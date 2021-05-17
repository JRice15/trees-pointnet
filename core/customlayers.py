import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model
from tensorflow.keras import backend as K
from tensorflow.keras import layers

from core import args

class TNet(keras.layers.Layer):
    """
    Tranformation Network, from B*InChannels to B*K*K
    """

    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__(self, **kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        w_init = tf.constant_initializer(0.0)
        self.w = tf.Variable(
            initial_value=w_init((in_channels, out_channels*out_channels)),
            trainable=True
        )
        self.b = tf.Variable(
            initial_value=K.flatten(K.eye(out_channels, dtype=tf.float32)),
            trainable=True
        )
        self.reshape = layers.Reshape((out_channels, out_channels), name=self.name+"_reshape")

    def call(self, x):
        x = tf.matmul(x, self.w)
        x += self.b
        transformation_matrix = self.reshape(x)
        return transformation_matrix

    def get_config(self):
        config = super().get_config()
        config = config.update({
            "in_channels": self.in_channels,
            "out_channels": self.out_channels,
        })
        return config


class MatMul(layers.Layer):
    """
    didn't work in a lambda layer for some reason
    """

    def call(self, x):
        a, b = x
        return tf.matmul(a, b)


class RaggedMatMul(layers.Layer):
    """
    matrix multiply batch of ragged tensors with a batch of transformation
    matrices. a surprisingly tricky thing to do
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._supports_ragged_inputs = True
        self.batchsize = args.batchsize

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
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "batchsize": self.batchsize
        })


def Activation(actname, **kwargs):
    """
    args:
        actname: name of activation function
    """
    if args.ragged:
        return layers.TimeDistributed(layers.Activation(actname), **kwargs)
    else:
        return layers.Activation(actname, **kwargs)


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
