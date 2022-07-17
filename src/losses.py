import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras import backend as K
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from tensorflow import keras
import math

from scipy.optimize import linear_sum_assignment
from sklearn.metrics import pairwise_distances

from src import ARGS
from src.utils import scale_meters_to_0_1


def get_loss():
    """
    return tuple of (loss, list(metrics), error_func).
    loss is a keras loss, and metrics are keras metrics.
    error func is a function with this signature:
        error_func(predictions, gt_targets): -> scalar
    where scalar is an error metric (can be negative)
    """
    if ARGS.loss == "mmd":
        return max_mean_discrepancy()
    if ARGS.loss == "gridmse":
        return grid_mse()
    if ARGS.loss == "p2p":
        return point_to_point()

    raise ValueError("No loss for mode '{}'".format(ARGS.loss))

@tf.function
def tf_height_1_gaussian(x, center, sigma):
    """
    evaluate a scaled gaussian centered at `center` for point(s) `x`
    see function `height_1_gaussian` in src/utils.py
    """
    squared_dists = tf.reduce_sum((x - center) ** 2, axis=-1)
    exp_factor = -1 / (2 * sigma ** 2)
    exp = tf.exp( exp_factor * squared_dists)
    return exp

@tf.function
def tf_rasterize_pts_gaussian_blur(weighted_pts, gaussian_sigma, resolution=50):
    """
    rasterize weighted points to a grid, with a gaussian blur
    works for batched input only, ie weighted_pts has shape (batchsize, npoints, channels)
    args:
        weighted_pts: (x,y,weight) locations (all between zero and one)
        guassian_sigma: the stddev of the guassian blur (in 0-1 scale)
    returns:
        grid of shape (batchsize, resolution, resolution)
    """
    # shape codes: B=batchsize, R=resolution, N=num weighted points
    batchsize = tf.shape(weighted_pts)[0]

    x = tf.linspace(0.0, 1.0, resolution)
    y = tf.linspace(0.0, 1.0, resolution)
    x, y = tf.meshgrid(x, y)
    gridcoords = tf.cast(tf.stack([x,y], axis=-1), dtype=weighted_pts.dtype) # (R,R,2)
    # expand out to batch shape
    mults = [batchsize] + [1 for _ in gridcoords.shape]
    gridcoords = tf.tile(gridcoords[None,...], mults) # (B,R,R,2)
    gridcoords = tf.expand_dims(gridcoords, axis=-2) # (B,R,R,1,2)

    # weighted_pts shape now: (B,N,3) where 3 => x,y,conf
    weighted_pts = tf.expand_dims(weighted_pts, axis=1)
    weighted_pts = tf.expand_dims(weighted_pts, axis=1)
    # shape is now: (B,1,1,N,3)
    pts = weighted_pts[...,:2] # (B,1,1,N,2)
    weights = weighted_pts[...,-1] # (B,1,1,N)
    gridvals = tf_height_1_gaussian(gridcoords, pts, sigma=gaussian_sigma) # (B,R,R,N)
    gridvals = gridvals * weights # (B,R,R,N)
    if ARGS.gridmse_agg == "sum":
        gridvals = tf.reduce_sum(gridvals, axis=-1)
    elif ARGS.gridmse_agg == "max":
        gridvals = tf.reduce_max(gridvals, axis=-1)
    else:
        raise ValueError("Unknown rasterize_pts mode")
    return gridvals # (B,R,R)



def grid_mse():
    """
    inspired by https://www.osapublishing.org/optica/fulltext.cfm?uri=optica-5-4-458&id=385495
    """
    assert ARGS.gaussian_sigma is not None

    resolution = 128 // ARGS.subdivide

    scaled_sigma = scale_meters_to_0_1(ARGS.gaussian_sigma, ARGS.subdivide)

    @tf.function
    def loss(gt, pred):
        pred_grid = tf_rasterize_pts_gaussian_blur(pred, scaled_sigma, resolution=resolution)
        gt_grid = tf_rasterize_pts_gaussian_blur(gt, scaled_sigma, resolution=resolution)

        # "squared l2 norm" = mean squared error
        square_error_grid = (pred_grid - gt_grid) ** 2
        err = tf.reduce_mean(square_error_grid, axis=[1, 2]) # average over all grid cells
        return err
    
    return loss, None


def max_mean_discrepancy():
    """
    DeepLoco loss
    from https://www.biorxiv.org/content/10.1101/267096v1.full
    """
    assert ARGS.gaussian_sigma is not None

    scaled_sigma = scale_meters_to_0_1(ARGS.gaussian_sigma, ARGS.subdivide)

    @tf.function(experimental_relax_shapes=True)
    def mmd_loss_term(a, b):
        a = tf.expand_dims(a, axis=1)
        a_locs = a[...,:2]
        a_weights = a[...,2]

        b = tf.expand_dims(b, axis=2)
        b_locs = b[...,:2]
        b_weights = b[...,2]

        # a shape: (B,1,N_a,3)
        # b shape: (B,N_b,1,3)

        # 0-1 isvalid indicator weights
        diff_weights = a_weights * b_weights

        # get loss from gaussian kernel
        #   a-b will create ground truth difference matrix, (B, N_b, N_a, 2)
        #   losses shape: (B, N_b, N_a)
        losses = tf_height_1_gaussian(a_locs, b_locs, scaled_sigma)
        loss = K.sum(losses * diff_weights)
        return loss

    @tf.function
    def mmd_loss(y, x):
        """
        equation 4.
        y: targets, shape (B,MaxNumTrees,3) where 3 channels are x,y,isvalidbit
        x: model outputs, shape (B,N,3) where 3 channels are x,y,weight
        """
        # y loss has no derivatives for our model, so doesn't help us learn anything
        # y_loss = mmd_loss_term(y, y)
        xy_loss = mmd_loss_term(x, y)
        x_loss = mmd_loss_term(x, x)

        # eq. 4
        # loss = y_loss - (2 * xy_loss) + x_loss
        loss = (-2 * xy_loss) + x_loss
        return loss / 1_000 # scaling factor

    return mmd_loss, None



def point_to_point():
    """
    From P2PNet (Song et al 2021): https://github.com/TencentYoutuResearch/CrowdCounting-P2PNet
    """

    tf.config.experimental_run_functions_eagerly(True)

    class CustomMean(tf.keras.metrics.Metric):
        __name__ = "CustomMean"

        def __init__(self, name, **kwargs):
            super().__init__(name=name, **kwargs)
            self.total = 0
            self.count = 0

        def update_state(self, y, x):
            pass
    
        def my_update_state(self, values):
            values = values.numpy()
            self.total += values.sum()
            self.count += values.size
        
        def reset_states(self):
            self.total = 0
            self.count = 0
        
        def result(self):
            if self.count == 0:
                return 0
            return self.total / self.count


    cls_matched_metric = CustomMean(name="cls_m_loss")
    cls_unmatched_metric = CustomMean(name="cls_unm_loss")
    loc_metric = CustomMean(name="loc_loss")
    all_metrics = [cls_matched_metric, cls_unmatched_metric, loc_metric]

    if ARGS.p2p_minmax_reg > 0:
        minmax_metric = CustomMean(name="minmax_reg")
        all_metrics.append(minmax_metric)


    # scale conf weight by subdivide so it has consistent effect independant of subdivide
    # ex: assume a dist of 0.3 when subdiv is 1. A conf of 0.1 will result in final cost of 0.2 (= 0.3 - 0.1), a 33% reduction
    #     however if subdiv is 3, dist is now 0.9. We still want this distance to be reduced the same amount, ie 33%, by the confidence. so we subtract conf * subdiv: 0.6 = 0.9 - (0.1 * 3)
    scaled_conf_weight = ARGS.p2p_conf_weight * ARGS.subdivide

    def matcher(pred, gt):
        """
        computes optimal matching assignment between the two
        args:
            preds: shape (B,npoints,3) where 3 -> (x, y, confidence)
            gt: shape (B,maxtrees,3) where 3 -> (x, y, isvalidtree)
        returns:
            matching: indexes of matched pred for each ground truth (B, npoints)
            ismatched: whether each pred is matched or not (B, npoints)
        """
        batchsize, maxtrees = tf.shape(gt).numpy()[:2]
        n_points = tf.shape(pred).numpy()[1]
        indices = np.zeros((batchsize, maxtrees), dtype=int)
        ismatched = np.zeros((batchsize, n_points), dtype=int)

        for idx,(pred_batch, gt_batch) in enumerate(zip(pred, gt)):
            # only use valid gt trees (which are always the first N in the array, so 
            # indices for valid trees won't change after this masking)
            gt_batch = gt_batch[gt_batch[:,2] > 0]
            n_gt = len(gt_batch)

            if n_gt > 0:
                # checknans(gt_batch, pred_batch)
                # calculate pairwise distances
                dists = pairwise_distances(gt_batch[:,:2], pred_batch[:,:2])
                # make high confidence things closer to everything, moderated by a weight.
                cost_matrix = dists - tf.expand_dims(scaled_conf_weight * pred_batch[:,2], axis=0)

                # find optimal assignment
                gt_inds, pred_inds = linear_sum_assignment(cost_matrix)
                # gt inds are always sorted 0-N, so not needed
                indices[idx,:n_gt] = pred_inds
                ismatched[idx,pred_inds] = 1
        
        indices = tf.stop_gradient( tf.constant(indices, dtype="int32") )
        ismatched = tf.stop_gradient( tf.constant(ismatched, dtype="int32") )
        return indices, ismatched

    # multiply all losses by this value for a nicer overall number
    LOSS_SCALE = 100
    # there are about 44 trees per full-sized patch on average
    trees_per_patch = 44 / (ARGS.subdivide ** 2)

    @tf.function
    def classification_loss(pred, ismatched):
        """
        Equation 4 in Song et al
        Make matched predictions have high confidence, and unmatched have low
        """
        # there are `trees_per_patch` trees expected on average, meaning that
        #   many preds will be matched on average each time. npoints - trees_per_patch
        #   will go unmatched. This factor makes it so they the matched and unmatched
        #   have the same total weight, before any reweighting by p2p_unmatched_weight
        n_output_points = pred.shape[1]
        unmatched_factor = trees_per_patch / (n_output_points - trees_per_patch)
        unmatched_factor *= ARGS.p2p_unmatched_weight

        confs = pred[...,2]
        ismatched = tf.cast(ismatched, K.floatx())
        # we want high confidence when matched to a gt, and low otherwise
        bce = K.binary_crossentropy(ismatched, confs)
        # scale only the loss at the unmatched points by p2p_unmatched_weight.
        # ismatched is acting as a true/false mask here
        matched_loss = (ismatched * bce)
        unmatched_loss = (unmatched_factor * (1-ismatched) * bce)
        # scale to make it nicer number
        matched_loss *= LOSS_SCALE
        unmatched_loss *= LOSS_SCALE
        # update metrics
        cls_matched_metric.my_update_state(matched_loss)
        cls_unmatched_metric.my_update_state(unmatched_loss)
        # sum elementwise (only one will be non-zero in each element
        loss = matched_loss + unmatched_loss
        return tf.reduce_mean(loss, axis=-1)

    @tf.function
    def location_loss(pred, gt, matching):
        """
        Equation 5 in Song et al
        Make matched predictions be close to their grount-truth matches
        """
        pred_locs_ordered = tf.gather(pred[...,:2], matching, axis=1, batch_dims=1)
        gt_locs = gt[...,:2]
        gt_isvalid = gt[...,2]
        # make distance be a consistent scale regardless of subdiv
        pred_locs_ordered /= ARGS.subdivide
        gt_locs /= ARGS.subdivide
        # squared distance (error) between gt and pred: (x1-x2)^2 + (y1-y2)^2
        error = (pred_locs_ordered - gt_locs) ** 2
        error = tf.reduce_sum(error, axis=-1)
        # average over valid matches in each example
        total = tf.reduce_sum(error * gt_isvalid, axis=-1)
        count_valid = tf.reduce_sum(gt_isvalid, axis=-1)
        # handle div by zero when no valid gt trees in an example
        loss = tf.where(count_valid > 0, total / count_valid, 0.0)
        loss *= ARGS.p2p_loc_weight * LOSS_SCALE # to make it a more convenient number
        # update metric
        loc_metric.my_update_state(loss)
        return loss

    def p2p_loss(gt, pred):
        """
        Equation 6 in Song et al
        """
        # get matching (no gradient there)
        matching, ismatched = matcher(pred, gt)
        # compute losses
        cls_loss = classification_loss(pred, ismatched)
        loc_loss = location_loss(pred, gt, matching)
        # final equation
        total_loss = cls_loss + loc_loss
        # regularize not all confs being the same
        if ARGS.p2p_minmax_reg > 0:
            conf = pred[:,2]
            minmax_loss = ARGS.p2p_minmax_reg * (1 + tf.reduce_min(conf) - tf.reduce_max(conf))
            total_loss += minmax_loss
            minmax_metric.my_update_state(minmax_loss)
        return total_loss

    return p2p_loss, all_metrics

