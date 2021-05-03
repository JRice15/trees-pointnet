import h5py
import keras
import numpy as np
import tensorflow as tf
from keras import Model
from keras import backend as K
from keras import layers
from keras.optimizers import Adam



def get_loss(args):
    """
    return tuple of (loss, list(metrics))
    """
    mode = args.mode.lower()
    if mode == "pointwise-treetop":
        return pointwise_treetop(args)
    if mode == "count":
        return keras.losses.mse, [keras.metrics.mse]

    raise ValueError("No loss for mode '{}'".format(name))


def pointwise_treetop(args):
    """
    squared xy distance of each point to closest ground truth target, weighted by 
    """
    assert args.dist_weight is not None

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

    @tf.function
    def dist_loss(x, y_locs):
        x_locs = x[:,:,:2]
        x_weights = x[:,:,2]
        sqr_dists = tf.map_fn(handle_batch, (x_locs, y_locs), 
            # fn_output_signature=tf.RaggedTensorSpec(shape=(None,), dtype=tf.float32))
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
        print("\n\nloss")
        print(x)
        print(y_locs)
        tf.print(x)
        tf.print(y_locs)
        return args.dist_weight * dist_loss(x, y_locs) + \
               (1-args.dist_weight) * count_loss(x, y_locs)
    
    return loss_func, [count_loss, dist_loss]
        


class RaggedMSE(keras.losses.Loss):
    @tf.function
    def call(self, y_true, y_pred):
        losses = tf.ragged.map_flat_values(
            keras.losses.mse, y_true, y_pred)
        return tf.reduce_mean(losses)





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
        pass

    args = A()

    args.dist_weight = 0.5

    # res = pointwise_treetop(args)(y, x)

    # print(res)

    def test(vals):
        x, y = vals
        print("\nf  x")
        tf.print(x)
        print("y")
        tf.print(y)
        return 1

    tf.ragged.map_flat_values(test, (x, y))

