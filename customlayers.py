import keras
import numpy as np
import tensorflow as tf
from keras import Model
from keras import backend as K
from keras import layers

def stack_ragged(tensors):
    """https://stackoverflow.com/questions/57346556/creating-a-ragged-tensor-from-a-list-of-tensors"""
    values = tf.concat(tensors, axis=0)
    lens = tf.stack([tf.shape(t, out_type=tf.int64)[0] for t in tensors])
    return tf.RaggedTensor.from_row_lengths(values, lens)

class MatMul(keras.layers.Layer):

    def __init__(self, batchsize, **kwargs):
        super().__init__(**kwargs)
        self.batchsize = batchsize
        self._supports_ragged_inputs = True

    def call(self, xs):
        a, b = xs
        # if isinstance(a, tf.RaggedTensor):
        #     print(a.values)
        #     y = tf.matmul(a.values, b)
        #     y = tf.RaggedTensor.from_nested_row_lengths(
        #         y,
        #         a.nested_row_lengths())
        # else:
        #     y = tf.matmul(a, b)

        vals = []
        for i in range(self.batchsize):
            y = tf.ragged.map_flat_values(tf.matmul, a[i], b[i])
            # y = tf.matmul(a.values[0], b)
            # y = tf.RaggedTensor.from_nested_row_lengths(
            #     y,
            #     a.nested_row_lengths())
            # y = tf.RaggedTensor.from_row_splits(y, [0])
            # y = tf.squeeze(y, axis=0)
            # y = tf.expand_dims(y, 0)
            print(y.shape, type(y))
            vals.append(y)

        out = stack_ragged(vals)

        # y = tf.ragged.map_flat_values(tf.matmul, a, b)
        print("out", out.shape)
        print("out", type(out))
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
