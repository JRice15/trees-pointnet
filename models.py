import keras
import numpy as np
import tensorflow as tf
from keras import Model
from keras import backend as K
from keras import layers


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
            initial_value=np.eye(out_channels, out_channels, dtype=np.float32).flatten(),
            trainable=True
        )
        self.reshape = layers.Reshape((out_channels, out_channels))

    def call(self, x):
        x = tf.matmul(x, self.w)
        x += self.b
        x = self.reshape(x)
        return x

    def get_config(self):
        config = super().get_config()
        config = update({
            "in_channels": self.in_channels,
            "K": self.out_channels,
        })
        return config



def pointnet_conv(outchannels, kernel_size):
    """
    returns callable pipeline of conv, batchnorm, and relu
    """
    def conv_op(x):
        x = layers.Conv2D(outchannels, kernel_size=kernel_size, padding="valid")(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        return x
    return conv_op


def pointnet_transform(x_in, kind):
    """
    args:
        x_in: (B,N,K) or (B,N,1,K), corresponding to kind
        kind: 'input' or 'feature'
    returns:
        (B,N,K)
    """
    assert kind in ("input", "feature")
    inputkind = (kind == "input")

    x = x_in

    if inputkind:
        batchsize, npoints, nattributes = x_in.shape
        kernel1shape = (1,nattributes)
        x = layers.Reshape((npoints, nattributes, 1))(x)
    else:
        batchsize, npoints, _, nattributes = x_in.shape
        kernel1shape = (1,1)

    x = pointnet_conv(64, kernel_size=kernel1shape)(x)
    x = pointnet_conv(128, kernel_size=1)(x)
    x = pointnet_conv(1024, kernel_size=1)(x)

    x = layers.MaxPool2D((npoints, 1))(x)
    x = layers.Flatten()(x)

    x = layers.Dense(512)(x)
    x = layers.Dense(256)(x)

    trans_matrix = TNet(256, nattributes)(x)
    
    if not inputkind:
        # squeeze out size-1 dimension
        x_in = layers.Reshape((npoints, nattributes))(x_in)
    
    # apply transformation matrix
    transformation = layers.Lambda(lambda x_in: tf.matmul(x_in, trans_matrix))
    x_out = transformation(x_in)

    print(x_out.shape)
    return x_out

    

def pointnet(npoints, nattributes):
    """
    args:
        npoints: number of points per patch
        nattributes: number of attributes per point (x,y,z, r,g,b, etc)
    """

    inpt = layers.Input((npoints, nattributes))

    x = inpt
    # input transform
    x = pointnet_transform(x, kind="input")

    # add channels
    x = layers.Reshape((npoints, nattributes, 1))(x)

    # mlp 1
    x = pointnet_conv(64, (1,nattributes))(x)
    x = pointnet_conv(64, 1)(x)

    # feature transform
    x = pointnet_transform(x, kind="feature")

    # expand dims
    batchsize, npoints, nattributes = x.shape
    x = layers.Reshape((npoints, 1, nattributes))(x)

    # mlp 2
    x = pointnet_conv(64, 1)(x)
    x = pointnet_conv(128, 1)(x)
    x = pointnet_conv(1024, 1)(x)

    # remove size-1 dim
    x = layers.Reshape((npoints, 1024))(x)

    # symmetric function: max pool
    x = layers.GlobalMaxPool1D()(x)

    # output flow
    x = layers.Dense(512, )

    model = Model(inpt, x)

    return model



