import h5py
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras import backend as K
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from tensorflow import keras
import math


def get_loss(ARGS):
    """
    return tuple of (loss, list(metrics), error_func).
    loss is a keras loss, and metrics are keras metrics.
    error func is a function with this signature:
        error_func(predictions, gt_targets): -> scalar
    where scalar is an error metric (can be negative)
    """
    if ARGS.mode == "pwtt":
        if ARGS.ragged:
            return ragged_pointwise_treetop(ARGS)
        else:
            return nonrag_pointwise_treetop(ARGS)
    if ARGS.mode == "count":
        return keras.losses.mse, [keras.metrics.mse]
    if ARGS.mode in ["mmd", "pwmmd"]:
        return max_mean_discrepancy(ARGS)

    raise ValueError("No loss for mode '{}'".format(ARGS.mode))




def max_mean_discrepancy(ARGS):
    """
    DeepLoco loss
    from https://www.biorxiv.org/content/10.1101/267096v1.full
    """
    assert ARGS.mmd_sigma is not None

    exp_constant = 4 * ARGS.mmd_sigma**2
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

    exp_constant = ARGS.mmd_sigma
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
        kernel_constant = (8 * math.pi**2 * ARGS.mmd_sigma**2) ** -0.5
    elif ARGS.mmd_kernel == "laplacian":
        kernel = laplacian_kernel_func
        kernel_constant = (2 * ARGS.mmd_sigma) ** -0.5
    else:
        raise ValueError("Unknown mmd kernel '{}' to mmd loss".format(ARGS.mmd_kernel))

    @tf.function
    def mmd_loss_term(a, b):
        a = tf.expand_dims(a, axis=1)
        a_locs = a[...,:2]
        a_weights = a[...,2]

        b = tf.expand_dims(b, axis=2)
        b_locs = b[...,:2]
        b_weights = b[...,2]

        # ground truth difference matrix, (B,MaxNumTrees,MaxNumTrees,3)
        # get xy-diffs, and 0,1 weights
        diffs = a_locs - b_locs
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
        x: model outputs, shape (B,MaxNumTrees,3) where 3 channels are x,y,weight
        """
        y_loss = mmd_loss_term(y, y)
        xy_loss = mmd_loss_term(x, y)
        x_loss = mmd_loss_term(x, x)

        # eq. 4. I pulled out the kernel constant from each K, and just multiply the final result (not that it really matters, as a constant in a loss term)
        loss = y_loss - (2 * xy_loss) + x_loss
        return kernel_constant * loss / 10_000 # scaling factor

    return mmd_loss, None



def nonrag_pointwise_treetop(ARGS):
    """
    squared xy distance of each point to closest ground truth target, weighted by tree/not-tree classification
    """
    assert ARGS.dist_weight is not None
    assert ARGS.dist_weight >= 0
    assert ARGS.dist_weight <= 1

    @tf.function
    def handle_example(inpts):
        """
        handles one element from a batch
        """
        x_locs, y = inpts
        y_locs = y[...,:2]
        y_isvalid = y[...,2:]

        # slightly hacky. add 10 to tree coords if y is not valid. since data is scaled 0-1, invalid trees will never be
        # the closest when the min happens later
        y_locs = y_locs + (10 * (1-y_isvalid))

        # expand to shape (1, npoints, 2), where 2 is for xy coords
        x_locs = tf.expand_dims(x_locs, 0)
        # expand to shape (ngroundtruth, 1, 2), where 2 is for xy coords
        y_locs = tf.expand_dims(y_locs, 1)
        # automatically expands to distance-matrix-esque of shape (ngroundtruth, npoints, 2))
        diff = x_locs - y_locs
        # find squared euclidean distance: (x2 - x2)^2 + (y2 - y1)^2
        sqr_dist = K.sum(diff ** 2, axis=-1)
        # minimize over ngroundtruth axis, so we get min distance for each point, not each gt tree
        dist2closest = K.min(sqr_dist, axis=0)
        return dist2closest

    @tf.function
    def dist_loss(y, x):
        """
        minimum squared distance from each x to each y, weighted by x weights
        """
        x_locs = x[:,:,:2]
        x_weights = x[:,:,2]
        sqr_dists = tf.vectorized_map(
            handle_example,
            [x_locs, y],
            fallback_to_while_loop=False
        )
        weighted_dists = sqr_dists * x_weights
        return ARGS.dist_weight * K.mean(weighted_dists)

    @tf.function
    def count_loss(y, x):
        """
        error between actual count of trees and sum of x weights
        """
        x_weights = x[:,:,2]
        y_is_valid = y[:,:,2]
        tree_counts = K.sum(y_is_valid, axis=-1) # per batch
        predicted_counts = K.sum(x_weights, axis=-1)
        return (1 - ARGS.dist_weight) * keras.losses.huber(tree_counts, predicted_counts)

    @tf.function
    def loss_func(y, x):
        loss = dist_loss(y, x)
        loss += count_loss(y, x)
        return loss

    return loss_func, [dist_loss, count_loss]


def ragged_pointwise_treetop(ARGS):
    """
    NOTE: this code may not work
    """
    assert ARGS.dist_weight is not None

    def x_handler(y_row):
        @tf.function
        def f(x):
            """
            min squared distance between any yrow point and the provided x
            """
            return K.min(K.sum((x - y_row) ** 2, axis=-1))
        return f

    @tf.function
    def handle_batch(inpt):
        """
        for each x point in this batch, find the squared distance to the closest y point
        """
        x, y = inpt
        out = tf.vectorized_map(x_handler(y), x)
        return out

    def dist_loss(x, y_locs):
        x_locs = x[:,:,:2]
        x_weights = x[:,:,2]
        sqr_dists = tf.map_fn(handle_batch, (x_locs, y_locs), 
            fn_output_signature=tf.RaggedTensorSpec(shape=(None,), dtype=tf.float32) # only tf > 2.2
            # dtype=tf.RaggedTensorSpec(shape=(None,), dtype=tf.float32) # tf 2.2
        )

        weighted_dists = sqr_dists * x_weights
        # mean loss per point, not mean of batch means. ie, losses for each batch is weighted by number of points
        return K.mean(weighted_dists)

    @tf.function
    def count_loss(x, y_locs):
        x_weights = x[:,:,2]
        # negative y -> not a tree
        y_is_positive = tf.sign(K.sum(y_locs, axis=-1) + K.epsilon())
        tree_count = K.sum(tf.nn.relu(y_is_positive), axis=-1) # per batch
        predicted_counts = K.sum(x_weights, axis=-1)
        # batchwise squared error between predicted and actual tree count. ie batch loss is not weighted
        return K.mean((tree_count - predicted_counts) ** 2)

    def loss_func(y_locs, x):
        loss = ARGS.dist_weight * dist_loss(x, y_locs)
        loss += (1 - ARGS.dist_weight) * count_loss(x, y_locs)
        return loss
    
    return loss_func, None
        




if __name__ == "__main__":

    # testing pointwise treetop

    x = [
        [[1, 1, 0.8], [0, 2, 0.2]],
        [[0, 0, 1.0], [0, 1, 0]],
        [[0, 0, 0.0], [0, 1, 1.0]]
    ]
    x = tf.constant(x, dtype=tf.float32)


    # y = [
    #     [[0, 0, 1], [0, 0, 1]],
    #     [[0, 1, 0], [0, 1.5, 1]],
    #     [[1, 0, 1], [100, 10, 0]],
    # ] # cls
    y = [
        [[0, 2.5, 1], [0, 0, 1]],
        [[0, 0, 1], [10, 10, 0]],
        [[0, 0, 1], [10, 10, 0]],
    ]
    y = tf.constant(y, dtype=tf.float32)


    class A:
        dist_weight = 0.5

    ARGS = A()
    res = nonrag_pointwise_treetop(ARGS)[0](y, x)

    print(res)


