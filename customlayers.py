import keras
import numpy as np
import tensorflow as tf
from keras import Model
from keras import backend as K
from keras import layers



class MatMul(keras.layers.Layer):
    """
    matrix multiply batch of ragged tensors with a batch of transformation
    matrices. a surprisingly tricky thing to do
    """

    def __init__(self, batchsize, **kwargs):
        super().__init__(**kwargs)
        self.batchsize = batchsize
        self._supports_ragged_inputs = True

    def call(self, xs):
        a, b = xs

        # manually split out batch and multiply individually
        vals = []
        for i in range(self.batchsize):
            y = tf.ragged.map_flat_values(tf.matmul, a[i], b[i])
            vals.append(y)

        # recombine into a ragged tensor
        values = tf.concat(vals, axis=0)
        lens = tf.stack([tf.shape(v, out_type=tf.int64)[0] for v in vals])
        out = tf.RaggedTensor.from_row_lengths(values, lens)
        return out



def ExpandDims(axis, **kwargs):
    return layers.Lambda(
        lambda x: tf.expand_dims(x, axis=axis),
        **kwargs
    )

def ReduceDims(axis, **kwargs):
    return layers.Lambda(
        lambda x: tf.squeeze(x, axis=axis),
        **kwargs
    )
