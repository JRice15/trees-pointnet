import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model
from tensorflow.keras import backend as K
from tensorflow.keras import layers

from src import ARGS, CUSTOM_LAYERS

class TNet(layers.Layer):
    """
    Tranformation Network, which learns a transformation matrix from input 
    from (B,inchannels) to (B,outsize,outsize)
    """

    def __init__(self, out_size, name, **kwargs):
        super().__init__(self, name=name, **kwargs)
        self.out_size = out_size

    def build(self, input_shape):
        in_channels = int(input_shape[-1])
        w_init = tf.constant_initializer(0.0)
        self.w = tf.Variable(
            initial_value=w_init((in_channels, self.out_size**2), dtype=K.floatx()),
            trainable=True,
            name=self.name+"_w"
        )
        self.b = tf.Variable(
            initial_value=K.flatten(K.eye(self.out_size, dtype=K.floatx())),
            trainable=True,
            name=self.name+"_b"
        )
        super().build(input_shape)

    def call(self, x):
        x = tf.matmul(x, self.w)
        x += self.b
        x = tf.reshape(x, (-1, self.out_size, self.out_size))
        return x

    def get_config(self):
        # super.get_config fails?
        config = super().get_config()
        return {
            "out_size": self.out_size,
            "name": self.name,
            **config
        }



class MatMul(layers.Layer):
    """
    didn't work in a lambda layer for some reason
    """

    def call(self, x):
        a, b = x
        return tf.matmul(a, b)



class Tile(layers.Layer):

    def __init__(self, num, axis, **kwargs):
        super().__init__(**kwargs)
        self.num = num
        self.axis = axis

    def build(self, input_shape):
        self.tilevec = [self.num if i == self.axis else 1 for i in range(len(input_shape))]

    def call(self, x):
        return tf.tile(x, self.tilevec)

    def get_config(self):
        config = super().get_config()
        return {
            "num": self.num,
            "axis": self.axis,
            **config
        }


CUSTOM_LAYERS["TNet"] = TNet
CUSTOM_LAYERS["Tile"] = Tile
CUSTOM_LAYERS["MatMul"] = MatMul


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
