import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras import backend as K
from tensorflow.keras import layers

from core import args


class MatMul(layers.Layer):
    """
    matrix multiply batch of ragged tensors with a batch of transformation
    matrices. a surprisingly tricky thing to do
    """

    def __init__(self, batchsize, **kwargs):
        super().__init__(**kwargs)
        self._supports_ragged_inputs = True
        self.batchsize = batchsize

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
        # print(out.shape)
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
