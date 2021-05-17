import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras import backend as K
from tensorflow.keras import layers
from tensorflow import keras

from core import customlayers
from core.data_loading import make_data_generators

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




def pointnet_conv(outchannels, kernel_size, name, bn=True, activation=True):
    """
    returns callable pipeline of conv, batchnorm, and relu
    input: (B,N,1,K)
    """
    def op(x):
        x = layers.TimeDistributed(
                layers.Conv1D(outchannels, kernel_size=kernel_size, padding="valid",
                    kernel_initializer="glorot_normal", name=name+"_inner"), name=name)(x)
        if bn:
            x = layers.TimeDistributed(
                    layers.BatchNormalization(name=name+"_bn_inner"), name=name+"_bn")(x)
        if activation:
            x = layers.TimeDistributed(
                    layers.ReLU(name=name+"_relu_inner"), name=name+"_relu")(x)
        return x
    return op


def pointnet_dense(outchannels, name, bn=True, activation=True):
    """
    returns callable pipeline of dense, batchnorm, and relu
    input: (B,N,K)
    """
    def op(x):
        x = layers.Dense(outchannels, kernel_initializer="glorot_normal", name=name)(x)
        if bn:
            x = layers.BatchNormalization(name=name+"_bn")(x)
        if activation:
            x = layers.ReLU(name=name+"_relu")(x)
        return x
    return op


def pointnet_transform(x_in, batchsize, kind):
    """
    args:
        x_in: (B,N,1,K)
        kind: 'input' or 'feature'
    returns:
        x_out: (B,N,K)
        transformation matrix: (K,K)
    """
    assert kind in ("input", "feature")
    prefix = kind + "_transform_"

    batchsize, ragged_npoints, _, nattributes = x_in.shape

    x = x_in

    x = pointnet_conv(64, kernel_size=1, name=prefix+"conv1")(x)
    x = pointnet_conv(128, kernel_size=1, name=prefix+"conv2")(x)
    x = pointnet_conv(1024, kernel_size=1, name=prefix+"conv3")(x)

    x = customlayers.ReduceDims(axis=2)(x) # (B,N,K)
    x = layers.GlobalMaxPool1D(name=prefix+"maxpool")(x) # (B,K)

    x = pointnet_dense(512, name=prefix+"dense1")(x)
    x = pointnet_dense(256, name=prefix+"dense2")(x)

    trans_matrix = TNet(256, nattributes, name=prefix+"tnet")(x)
    
    x_in = customlayers.ReduceDims(axis=2)(x_in)
    
    # apply transformation matrix
    x_out = customlayers.MatMul(batchsize=batchsize, 
                name=prefix+"matmul")([x_in, trans_matrix])

    return x_out, trans_matrix


def seg_output_flow(local_features, global_feature, outchannels):
    """
    Base segmentation output flow
    Returns:
        per-point feature vectors: (B,N,outchannels)
    """
    # concat local and global
    global_feature = layers.Reshape((1,1,1024), name="expand_global_feature_reshape")(global_feature)
    # global_feature = layers.Lambda(
    #     lambda feature: tf.tile(feature, [1, npoints, 1, 1]),
    #     name="global_feature_tile")(global_feature)
    
    full_features = layers.Concatenate(axis=-1)([local_features, global_feature])

    # print("shape of local, global, full:", local_features.shape, global_feature.shape, full_features.shape)

    # output flow
    x = full_features
    x = pointnet_conv(512, 1, name="outmlp_conv1")(x)
    x = pointnet_conv(256, 1, name="outmlp_conv2")(x)
    x = pointnet_conv(128, 1, name="outmlp_conv3")(x)
    x = pointnet_conv(128, 1, name="outmlp_conv4")(x)

    x = pointnet_conv(outchannels, 1, name="outmlp_conv_final", 
                      bn=False, activation=False)(x)
    # x = layers.Reshape((npoints, outchannels), name="output_squeeze")(x)
    x = customlayers.ReduceDims(axis=2, name="outmlp_squeeze")(x)
    
    return x

def cls_output_flow(global_feature, outchannels, dropout=0.3):
    """
    Base global classification output flow
    returns:
        global feature vector: (B,outchannels)
    """
    x = global_feature

    x = pointnet_dense(512, "outmlp_dense1")(x)
    if dropout > 0:
        x = layers.Dropout(dropout)(x)
    x = pointnet_dense(256, "outmlp_dense2")(x)
    if dropout > 0:
        x = layers.Dropout(dropout)(x)

    x = pointnet_dense(outchannels, "outmlp_dense3", bn=False, 
            activation=False)(x)

    return x


def pointnet(args, nattributes, output_features, reg_weight=0.001):
    """
    args:
        nattributes: number of attributes per point (x,y,z, r,g,b, etc)
        output_features: num features per output point
    """

    inpt = layers.Input((None, nattributes), ragged=True, batch_size=args.batchsize) # (B,N,K)

    x = inpt
    x = customlayers.ExpandDims(axis=2, name="add_channels_1")(x) # (B,N,1,K)

    # input transform
    x, inpt_trans_matrix = pointnet_transform(x, batchsize=args.batchsize, kind="input") # (B,N,K)
    x = customlayers.ExpandDims(axis=2, name="add_channels_2")(x) # (B,N,1,K)

    # mlp 1
    x = pointnet_conv(64, 1, name="mlp1_conv1")(x)
    x = pointnet_conv(64, 1, name="mlp1_conv2")(x)

    # feature transform
    x, feat_trans_matrix = pointnet_transform(x, batchsize=args.batchsize, kind="feature") # (B,N,K)
    x = customlayers.ExpandDims(axis=2, name="add_channels_3")(x) # (B,N,1,K)

    local_features = x

    # mlp 2
    x = pointnet_conv(64, 1, name="mlp2_conv1")(x)
    x = pointnet_conv(128, 1, name="mlp2_conv2")(x)
    x = pointnet_conv(1024, 1, name="mlp2_conv3")(x)

    # symmetric function: max pooling
    x = customlayers.ReduceDims(axis=2, name="remove_channels")(x) # (B,N,K)
    x = layers.GlobalMaxPool1D(name="global_feature_maxpool")(x) # (B,K)

    global_feature = x

    # output flow
    if args.output_type == "seg":
        output = seg_output_flow(local_features, global_feature, output_features)
    elif args.output_type == "cls":
        output = cls_output_flow(global_feature, output_features)
    
    if args.mode == "pointwise-treetop":
        # add input xy locations to each point
        output = layers.TimeDistributed(
                layers.Activation("sigmoid"), name="pwtt-sigmoid")(output) # limit to 0 to 1
        output = layers.Concatenate(axis=-1, name="pwtt-concat_inpt")([inpt, output])

    model = Model(inpt, output)

    # feature transformation matrix orthogonality loss
    dims = feat_trans_matrix.shape[1]
    ortho_diff = tf.matmul(feat_trans_matrix,
                    tf.transpose(feat_trans_matrix, perm=[0,2,1]))
    ortho_diff -= tf.constant(tf.eye(dims), dtype=tf.float32)
    ortho_loss = reg_weight * tf.nn.l2_loss(ortho_diff)
    model.add_loss(ortho_loss)
    model.add_metric(ortho_loss, name="ortho_loss", aggregation="mean")


    return model




if __name__ == "__main__":

    class args:
        output_type = "cls"
        mode = "count"
        batchsize = 5

    model = pointnet(args, 3, 1)

    traingen, _, _ = make_data_generators(args.mode, args.batchsize)

    x, y = traingen[3]

    model.compile()

    out = model.predict(x)

    print(out)



