import h5py
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras import backend as K
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from tensorflow import keras


def get_loss(ARGS):
    """
    return tuple of (loss, list(metrics), error_func).
    loss is a keras loss, and metrics are keras metrics.
    error func is a function with this signature:
        error_func(predictions, gt_targets): -> scalar
    where scalar is an error metric (can be negative)
    """
    mode = ARGS.mode.lower()
    if mode == "pointwise-treetop":
        if ARGS.ragged:
            return ragged_pointwise_treetop(ARGS)
        else:
            return nonrag_pointwise_treetop(ARGS)
    if mode == "count":
        return keras.losses.mse, [keras.metrics.mse], lambda x,y: y - x

    raise ValueError("No loss for mode '{}'".format(mode))


def nonrag_pointwise_treetop(ARGS):
    """
    squared xy distance of each point to closest ground truth target, weighted by tree/not-tree classification
    """
    assert ARGS.dist_weight is not None

    @tf.function
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

    @tf.function
    def loss_func(y_locs, x):
        loss = ARGS.dist_weight * dist_loss(x, y_locs)
        loss += (1 - ARGS.dist_weight) * count_loss(x, y_locs)
        return loss
    
    return loss_func, None, None


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
    
    return loss_func, None, None
        




if __name__ == "__main__":

    # testing pointwise treetop

    x = [
        [[0, 1, 0.8], [0, 2, 0.2]],
        [[0, 0, 1.0]],
        [[0, 0, 0.0], [0, 1, 1.0], [0, 2, 0.3], [0, 3, 0.1]]
    ]
    x = tf.ragged.constant(x, ragged_rank=1, dtype=tf.float32)


    y = [
        [[0, 0], [0, 0], [-100, -100]],
        [[0, 1], [0, 3], [10, 10]],
        [[1, 0], [100, 10], [0, 4]],
    ] # cls
    y = tf.constant(y, dtype=tf.float32)


    class A:
        dist_weight = 0.5

    ARGS = A()
    res = pointwise_treetop(ARGS)[0](y, x)

    print(res)


