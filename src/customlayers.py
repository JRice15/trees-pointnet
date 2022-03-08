import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model
from tensorflow.keras import backend as K
from tensorflow.keras import layers

from src import ARGS

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
            initial_value=w_init((in_channels, self.out_size**2)),
            trainable=True,
            name=self.name+"_w"
        )
        self.b = tf.Variable(
            initial_value=K.flatten(K.eye(self.out_size, dtype=tf.float32)),
            trainable=True,
            name=self.name+"_b"
        )

    def call(self, x):
        x = tf.matmul(x, self.w)
        x += self.b
        x = tf.reshape(x, (-1, self.out_size, self.out_size))
        return x

    def get_config(self):
        # super.get_config fails?
        #config = super().get_config()
        return {
            "out_size": self.out_size,
            "name": self.name,
            #**config
        }


class ConcatGrid(layers.Layer):
    """
    unused
    """

    def __init__(self, gridsize, depth_factor, **kwargs):
        super().__init__(self, **kwargs)
        self.gridsize = gridsize
        self.depth_factor = depth_factor
        # self.gridconv = layers.Conv2D(depth_factor, 3, padding="same")
    
    def build(self, input_shape):
        xgrid, ygrid = np.meshgrid(
            np.linspace(0, 1, self.gridsize),
            np.linspace(0, 1, self.gridsize)
        )
        grid = np.stack([xgrid, ygrid], axis=-1)
        self.grid = tf.constant(grid, shape=[1,self.gridsize,self.gridsize,2], dtype=K.floatx())
        print("grid:", self.grid.shape)

    def call(self, x):
        # add batch size to grid
        batchsize = tf.shape(x)[0]
        tilevec = [batchsize, 1, 1, self.depth_factor]
        grid = tf.tile(self.grid, tilevec)
        return tf.concat([grid, x], axis=-1)

    def get_config(self):
        config = {
            "gridsize": self.gridsize,
        }
        return config

class GatherTopK(layers.Layer):
    """
    DEPRECATED
    Does not work, I think
    call signiture:
        l = GatherTopK(...)
        top_n_from_data = l([data, confidences])
    """

    def __init__(self, k, name, **kwargs):
        super().__init__(name=name, **kwargs)
        self.k = k

    def call(self, x):
        data, confidences = x
        tf.print(tf.shape(data), tf.shape(confidences))
        values, indices = tf.nn.top_k(confidences, k=self.k)
        return tf.gather(data, indices, axis=1)

    def get_config(self):
        config = super().get_config()
        return {
            "k": self.k,
            **config
        }


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
        self.batchsize = ARGS.batchsize

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
        return {
            "batchsize": self.batchsize,
            **config
        }


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


def Activation(actname, **kwargs):
    """
    ARGS:
        actname: name of activation function
    """
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
