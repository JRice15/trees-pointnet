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
from src.utils import scaled_0_1


def get_loss():
    """
    return tuple of (loss, list(metrics), error_func).
    loss is a keras loss, and metrics are keras metrics.
    error func is a function with this signature:
        error_func(predictions, gt_targets): -> scalar
    where scalar is an error metric (can be negative)
    """
    if ARGS.loss == "treetop":
        assert ARGS.output_mode in ("seg", "dense")
        raise NotImplementedError()
        # if ARGS.ragged:
        #     return ragged_pointwise_treetop()
        # else:
        #     return nonrag_pointwise_treetop()
    if ARGS.loss == "mmd":
        assert ARGS.output_mode in ("seg", "dense")
        return max_mean_discrepancy()
    if ARGS.loss == "gridmse":
        assert ARGS.output_mode in ("seg", "dense")
        return grid_mse()
    if ARGS.loss == "count":
        assert ARGS.output_mode == "count"
        return keras.losses.mse, [keras.metrics.mse]

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
        weighted_pts: (x,y,weight) locations
        guassian_sigma: the stddev of the guassian blur
    """
    batchsize = tf.shape(weighted_pts)[0]

    x = tf.linspace(0.0, 1.0, resolution)
    y = tf.linspace(0.0, 1.0, resolution)
    x, y = tf.meshgrid(x, y)
    gridcoords = tf.cast(tf.stack([x,y], axis=-1), dtype=weighted_pts.dtype)
    # expand out to batch shape
    mults = [batchsize] + [1 for i in gridcoords.shape]
    gridcoords = tf.tile(gridcoords[None,...], mults)
    gridcoords = tf.expand_dims(gridcoords, axis=-2)

    weighted_pts = tf.expand_dims(weighted_pts, axis=-3)
    weighted_pts = tf.expand_dims(weighted_pts, axis=-3)
    pts = weighted_pts[...,:2]
    weights = weighted_pts[...,-1]
    gridvals = tf_height_1_gaussian(gridcoords, pts, sigma=gaussian_sigma)
    gridvals = gridvals * weights
    if ARGS.grid_agg == "sum":
        gridvals = tf.reduce_sum(gridvals, axis=-1)
    elif ARGS.grid_agg == "max":
        gridvals = tf.reduce_max(gridvals, axis=-1)
    else:
        raise ValueError("Unknown rasterize_pts mode")
    return gridvals



def grid_mse():
    """
    inspired by https://www.osapublishing.org/optica/fulltext.cfm?uri=optica-5-4-458&id=385495
    """
    assert ARGS.gaussian_sigma is not None

    resolution = 50 # TODO grid resolution

    scaled_sigma = scaled_0_1(ARGS.gaussian_sigma)

    @tf.function
    def loss(gt, pred):
        pred_grid = tf_rasterize_pts_gaussian_blur(pred, scaled_sigma, resolution=resolution)
        gt_grid = tf_rasterize_pts_gaussian_blur(gt, scaled_sigma, resolution=resolution)

        # "squared l2 norm" = mean squared error
        square_error_grid = (pred_grid - gt_grid) ** 2
        err = tf.reduce_mean(square_error_grid, axis=[1, 2])
        return err
    
    return loss, None


def max_mean_discrepancy():
    """
    DeepLoco loss
    from https://www.biorxiv.org/content/10.1101/267096v1.full
    """
    assert ARGS.gaussian_sigma is not None and ARGS.mmd_kernel is not None

    scaled_sigma = scaled_0_1(ARGS.gaussian_sigma)
    exp_constant = 4 * scaled_sigma**2
    @tf.function
    def gaussian_kernel_func(diffs):
        """
        equation 5 is general form, eq. 6 is this version.
        diffs is x_locations - y_locations
        """
        sqr_dists = K.sum(K.square(diffs), axis=-1)
        exponent = -sqr_dists / exp_constant
        exponent = K.clip(exponent, None, 64)
        return K.exp(exponent)

    exp_constant = scaled_sigma
    @tf.function
    def laplacian_kernel_func(diffs):
        """
        eq. 5 is general form, eq 10 specifically
        """
        l1_dists = tf.norm(diffs, ord=1, axis=-1)
        exponent = K.clip(-l1_dists / exp_constant, None, 64)
        return K.exp(exponent)

    # get correct kernel function
    if ARGS.mmd_kernel == "gaussian":
        kernel = gaussian_kernel_func
        kernel_constant = (8 * math.pi**2 * scaled_sigma**2) ** -0.5
    elif ARGS.mmd_kernel == "laplacian":
        kernel = laplacian_kernel_func
        kernel_constant = (2 * scaled_sigma) ** -0.5
    else:
        raise ValueError("Unknown mmd kernel '{}' to mmd_loss".format(ARGS.mmd_kernel))

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

        # ground truth difference matrix, (B,N_b,N_a,3)
        diffs = a_locs - b_locs
        # 0-1 isvalid indicator weights
        diff_weights = a_weights * b_weights
        # get loss from kernel function
        losses = kernel(diffs)
        loss = K.sum(losses * diff_weights)
        return loss

    @tf.function
    def mmd_loss(y, x):
        """
        equation 4.
        y: targets, shape (B,MaxNumTrees,3) where 3 channels are x,y,isvalidbit
        x: model outputs, shape (B,N,3) where 3 channels are x,y,weight
        """
        y_loss = mmd_loss_term(y, y)
        xy_loss = mmd_loss_term(x, y)
        x_loss = mmd_loss_term(x, x)

        # eq. 4. I pulled out the kernel constant from each K, and just multiply the final result (not that it really matters, as a constant in a loss term)
        loss = y_loss - (2 * xy_loss) + x_loss
        return kernel_constant * loss / 10_000 # scaling factor

    return mmd_loss, None




def point_to_point():
    """
    From P2PNet (Song et al 2021): https://github.com/TencentYoutuResearch/CrowdCounting-P2PNet
    """

def matcher(pred, gt):
    """
    args:
        preds: shape (B,npoints,3) where 3 -> (x, y, confidence)
        gt: shape (B,maxtrees,3) where 3 -> (x, y, isvalidtree)
    returns:
        matching: indexes of matched pred for each ground truth (B, )
    """
    batchsize, maxtrees = tf.shape(gt)[:2]
    n_points = tf.shape(pred)[1]
    indices = np.zeros((batchsize, maxtrees), dtype=int)
    is_match = np.zeros((batchsize, n_points))

    for idx,(batch_pred, batch_gt) in enumerate(zip(pred, gt)):
        # only use valid gt trees (which are always the first N in the array, so 
        # indices for valid trees won't change after this masking)
        batch_gt = batch_gt[batch_gt[:,2] > 0]
        n_gt = len(batch_gt)

        # calculate pairwise distances
        dists = pairwise_distances(batch_gt[:,:2], batch_pred[:,:2])
        # make high confidence things closer to everything, moderated by a weight
        cost_matrix = dists - tf.expand_dims(ARGS.p2p_conf_weight * batch_pred[:,2], axis=0)

        # find optimal assignment
        gt_inds, pred_inds = linear_sum_assignment(cost_matrix)
        # gt inds are always sorted 0-N, so not needed
        indices[idx,:n_gt] = pred_inds
        is_match[idx,pred_inds] = 1.0
    
    return tf.constant(indices, dtype="int32"), tf.constant(is_match, dtype=K.floatx())

    @tf.function
    def classification_loss(pred, ismatched):
        """
        Equation 4 in Song et al
        Make matched predictions have high confidence, and unmatched have low
        """
        confs = pred[:,2]
        n_ismatched = tf.reduce_sum(ismatched, axis=-1)
        # everything is either matched or unmatched
        pos_elements = tf.math.log(confs * ismatched)
        neg_elements = tf.math.log((1 - confs) * (1 - ismatched))
        neg_elements *= ARGS.p2p_unmatched_weight
        # sum together elementwise. Only one of the two will be non-zero in each element
        all_elements = pos_elements + neg_elements
        # binary crossentropy loss
        return -tf.reduce_mean(tf.math.log(all_elements))
    
    @tf.function
    def location_loss(pred, gt, matching):
        """
        Equation 5 in Song et al
        Make matched predictions be close to their grount-truth matches
        """
        pred_locs_ordered = tf.gather(pred[:,:2], matching, axis=1)
        gt_locs = gt[:,:2]
        gt_isvalid = gt[:,2]
        # squared distance (error) between gt and pred
        error = (pred_locs_ordered - gt_locs) ** 2
        # average over valid matches
        error = (error * gt_isvalid) / tf.reduce_sum(gt_isvalid, axis=-1)
        return error


    @tf.function
    def p2p_from_matching(pred, gt, matching, ismatched):
        """
        Equation 6 in Song et al
        """
        return classification_loss(pred, ismatched) + \
            ( ARGS.p2p_loc_weight * location_loss(pred, gt, matching) )

    def p2p_loss(gt, pred):
        # get matching (no gradient there)
        matching, ismatched = tf.stop_gradient( matcher(pred, gt) )
        # compute loss
        return p2p_from_matching(pred, gt, matching, ismatched)
