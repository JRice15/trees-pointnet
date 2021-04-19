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



def pointnet_conv(outchannels, kernel_size, name, bn=True, activation=True):
    """
    returns callable pipeline of conv, batchnorm, and relu
    """
    def op(x):
        x = layers.Conv2D(outchannels, kernel_size=kernel_size, padding="valid",
                          kernel_initializer="glorot_normal",
                          name=name)(x)
        if bn:
            x = layers.BatchNormalization(name=name+"_bn")(x)
        if activation:
            x = layers.ReLU(name=name+"_relu")(x)
        return x
    return op


def pointnet_dense(outchannels, name, bn=True, activation=True):
    """
    returns callable pipeline of dense, batchnorm, and relu
    """
    def op(x):
        x = layers.Dense(outchannels, kernel_initializer="glorot_normal",
                         name=name)(x)
        if bn:
            x = layers.BatchNormalization(name=name+"_bn")(x)
        if activation:
            x = layers.ReLU(name=name+"_relu")(x)
        return x
    return op


def pointnet_transform(x_in, kind):
    """
    args:
        x_in: (B,N,K) or (B,N,1,K), corresponding to kind
        kind: 'input' or 'feature'
    returns:
        x_out: (B,N,K)
        transformation matrix: (K,K)
    """
    assert kind in ("input", "feature")
    inputkind = (kind == "input")

    prefix = kind + "_transform_"

    x = x_in

    if inputkind:
        batchsize, npoints, nattributes = x_in.shape
        kernel1shape = (1,nattributes)
        x = layers.Reshape((npoints, nattributes, 1), name=prefix+"add_channels")(x)
    else:
        batchsize, npoints, _, nattributes = x_in.shape
        kernel1shape = (1,1)

    x = pointnet_conv(64, kernel_size=kernel1shape, name=prefix+"1")(x)
    x = pointnet_conv(128, kernel_size=1, name=prefix+"2")(x)
    x = pointnet_conv(1024, kernel_size=1, name=prefix+"3")(x)

    x = layers.MaxPool2D((npoints, 1), name=prefix+"maxpool")(x)
    x = layers.Flatten(name=prefix+"flatten")(x)

    x = layers.Dense(512, name=prefix+"dense1")(x)
    x = layers.Dense(256, name=prefix+"dense2")(x)

    trans_matrix = TNet(256, nattributes, name=prefix+"tnet")(x)
    
    if not inputkind:
        # squeeze out size-1 dimension
        x_in = layers.Reshape((npoints, nattributes), name=prefix+"squeeze")(x_in)
    
    # apply transformation matrix
    x_out = MatMul(name=prefix+"matmul")([x_in, trans_matrix])

    return x_out, trans_matrix

    

def pointnet(npoints, nattributes, reg_weight=0.001):
    """
    args:
        npoints: number of points per patch
        nattributes: number of attributes per point (x,y,z, r,g,b, etc)
    """

    inpt = layers.Input((npoints, nattributes))

    x = inpt
    # input transform
    x, inpt_trans_matrix = pointnet_transform(x, kind="input")

    # add channels
    x = layers.Reshape((npoints, nattributes, 1), name="add_channels_reshape")(x)

    # mlp 1
    x = pointnet_conv(64, (1,nattributes), name="mlp1_conv1")(x)
    x = pointnet_conv(64, 1, name="mlp1_conv2")(x)

    # feature transform
    x, feat_trans_matrix = pointnet_transform(x, kind="feature")

    # expand dims
    batchsize, npoints, nattributes = x.shape
    x = layers.Reshape((npoints, 1, nattributes), name="expand_dims_reshape")(x)

    local_features = x

    # mlp 2
    x = pointnet_conv(64, 1, name="mlp2_conv1")(x)
    x = pointnet_conv(128, 1, name="mlp2_conv2")(x)
    x = pointnet_conv(1024, 1, name="mlp2_conv3")(x)

    # remove size-1 dim
    x = layers.Reshape((npoints, 1024), name="premaxpool_reshape")(x)

    # symmetric function: max pool
    x = layers.GlobalMaxPool1D(name="global_feature_maxpool")(x)

    global_feature = x

    # concat local and global
    global_feature = layers.Reshape((1,1,1024), name="expand_global_feature_reshape")(x)
    global_feature = layers.Lambda(
        lambda feature: tf.tile(feature, [1, npoints, 1, 1]),
        name="global_feature_tile")(global_feature)
    
    full_features = layers.Concatenate(axis=-1)([local_features, global_feature])

    print("local, global, full:", local_features.shape, global_feature.shape, full_features.shape)

    # output flow
    x = full_features
    x = pointnet_conv(512, 1, name="outmlp_conv1")(x)
    x = pointnet_conv(256, 1, name="outmlp_conv2")(x)
    x = pointnet_conv(128, 1, name="outmlp_conv3")(x)
    x = pointnet_conv(128, 1, name="outmlp_conv4")(x)

    x = pointnet_conv(50, 1, name="outmlp_conv_final", 
                      bn=False, activation=False)(x)
    x = layers.Reshape((npoints, 50), name="output_squeeze")(x)
    output = x

    model = Model(inpt, output)

    # feature transformation matrix orthogonality loss
    dims = feat_trans_matrix.shape[1]
    ortho_diff = tf.matmul(feat_trans_matrix,
                    tf.transpose(feat_trans_matrix, perm=[0,2,1]))
    ortho_diff -= tf.constant(np.eye(dims), dtype=tf.float32)
    ortho_loss = reg_weight * tf.norm(ortho_diff, ord='euclidean')
    model.add_loss(ortho_loss)
    model.add_metric(ortho_loss, name="ortho_loss", aggregation="mean")


    return model



