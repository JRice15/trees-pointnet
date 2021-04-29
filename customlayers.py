import keras
import numpy as np
import tensorflow as tf
from keras import Model
from keras import backend as K
from keras import layers



class MatMul(keras.layers.Layer):
    """this operation didn't work in a lambda layer for some reason"""

    def call(self, xs):
        a, b = xs
        return tf.matmul(a, b)



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
