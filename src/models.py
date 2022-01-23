import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras import backend as K
from tensorflow.keras import layers
from tensorflow import keras

from src import customlayers, ARGS


def pointnet_conv(outchannels, kernel_size, name, strides=1, bn=True, activation=True,
        padding="valid"):
    """
    returns callable, which creates pipeline of conv, batchnorm, and relu
    input: (B,N,1,K)
    """
    layer_list = [
        layers.Conv1D(int(outchannels), kernel_size=kernel_size, padding=padding,
                    strides=strides, kernel_initializer="glorot_normal", name=name),
    ]
    if bn:
        layer_list.append(layers.BatchNormalization(name=name+"_bn"))
    if activation:
        layer_list.append(layers.ReLU(name=name+"_relu"))
    if ARGS.ragged:
        for i,v in enumerate(layer_list):
            layer_list[i] = layers.TimeDistributed(v, name=v.name+"_timedistrib")
    def op(x):
        for layer in layer_list:
            x = layer(x)
        return x
    return op


def pointnet_dense(outchannels, name, bn=True, activation=True):
    """
    returns callable, which creates pipeline of dense, batchnorm, and relu
    input: (B,N,K)
    """
    def op(x):
        x = layers.Dense(int(outchannels), kernel_initializer="glorot_normal", name=name)(x)
        if bn:
            x = layers.BatchNormalization(name=name+"_bn")(x)
        if activation:
            x = layers.ReLU(name=name+"_relu")(x)
        return x
    return op


def pointnet_transform(x_in, batchsize, kind):
    """
    ARGS:
        x_in: (B,N,1,K)
        kind: 'input' or 'feature'
    returns:
        x_out: (B,N,K)
        transformation matrix: (K,K)
    """
    assert kind in ("input", "feature")
    prefix = kind + "_transform_"

    # (B, N, 1, K)
    batchsize, _, _, nattributes = x_in.shape

    x = x_in

    x = pointnet_conv(64, kernel_size=1, name=prefix+"conv1")(x)
    x = pointnet_conv(128, kernel_size=1, name=prefix+"conv2")(x)
    x = pointnet_conv(1024, kernel_size=1, name=prefix+"conv3")(x)

    x = customlayers.ReduceDims(axis=2, name=prefix+"squeeze")(x) # (B,N,K)
    x = layers.GlobalMaxPool1D(name=prefix+"maxpool")(x) # (B,K)

    x = pointnet_dense(512, name=prefix+"dense1")(x)
    x = pointnet_dense(256, name=prefix+"dense2")(x)

    trans_matrix = customlayers.TNet(256, nattributes, name=prefix+"tnet")(x) # (B,(K*K))
    trans_matrix = layers.Reshape((nattributes, nattributes), 
                        name=prefix+"transmatrix_reshape")(trans_matrix) # (B,K,K)

    x_in = customlayers.ReduceDims(axis=2, name=prefix+"squeeze2")(x_in)
    
    # apply transformation matrix
    if ARGS.ragged:
        x_out = customlayers.RaggedMatMul(name=prefix+"matmul")([x_in, trans_matrix])
    else:
        x_out = customlayers.MatMul(name=prefix+"matmul")([x_in, trans_matrix])

    return x_out, trans_matrix


def seg_output_flow(local_features, global_feature, outchannels):
    """
    Base segmentation output flow
    Returns:
        per-point feature vectors: (B,N,outchannels)
    """
    # concat local and global
    global_feature = layers.Reshape((1,1,1024), name="expand_global_feature_reshape")(global_feature)
    if not ARGS.ragged:
        _, npoints, _, _ = local_features.shape
        global_feature = customlayers.Tile(npoints, axis=1)(global_feature)

    full_features = layers.Concatenate(axis=-1)([local_features, global_feature])

    # output flow
    x = full_features
    x = pointnet_conv(512, 1, name="outmlp_conv1")(x)
    x = pointnet_conv(256, 1, name="outmlp_conv2")(x)
    x = pointnet_conv(128, 1, name="outmlp_conv3")(x)
    x = pointnet_conv(128, 1, name="outmlp_conv4")(x)

    x = pointnet_conv(outchannels, 1, name="outmlp_conv_final",
                      bn=False, activation=False)(x)
    x = customlayers.ReduceDims(axis=2, name="outmlp_squeeze")(x)
    
    return x

def cls_output_flow(global_feature, outchannels):
    """
    Base global classification output flow
    returns:
        global feature vector: (B,outchannels)
    """
    x = global_feature

    x = pointnet_dense(512, "outmlp_dense1")(x)
    if ARGS.dropout > 0:
        x = layers.Dropout(ARGS.dropout, name="outmlp_dropout1")(x)
    x = pointnet_dense(256, "outmlp_dense2")(x)
    if ARGS.dropout > 0:
        x = layers.Dropout(ARGS.dropout, name="outmlp_dropout2")(x)

    x = pointnet_dense(outchannels, "outmlp_dense3", bn=False, 
            activation=False)(x)

    return x



def dense_output_flow_2(global_feature, out_npoints, out_channels):
    """
    Densely learning output points from the global feature alone
    args:
        global_feature
        out_npoints: number of points to output
        out_channels: number of channels per point (mode dependant)
    """
    x = global_feature

    input_size = x.shape[-1]
    target_size = out_npoints * out_channels
    intermediate_size = (input_size + target_size) // 2

    x = pointnet_dense(intermediate_size, "outmlp_dense1")(x)
    x = pointnet_dense(target_size, "outmlp_dense2")(x)
    x = layers.Reshape((out_npoints, out_channels))(x)
    x = pointnet_conv(out_channels, 1, bn=False, activation=False, name="out_channels_conv")(x)

    return x



def pointnet(inpt_shape, size_multiplier, output_channels, reg_weight=0.001):
    """
    args:
        output_channels: num features per output point
    """
    npoints, nattributes = inpt_shape
    inpt = layers.Input(inpt_shape, ragged=ARGS.ragged, 
                batch_size=ARGS.batchsize if ARGS.ragged else None) # (B,N,K)
    xy_locs = inpt[...,:2]

    x = inpt
    x = customlayers.ExpandDims(axis=2, name="add_channels_1")(x) # (B,N,1,K)

    if ARGS.use_tnet_1:
        # input transform
        x, inpt_trans_matrix = pointnet_transform(x, batchsize=ARGS.batchsize, kind="input") # (B,N,K)
        x = customlayers.ExpandDims(axis=2, name="add_channels_2")(x) # (B,N,1,K)

    # mlp 1
    x = pointnet_conv(64*size_multiplier, 1, name="mlp1_conv1")(x)
    x = pointnet_conv(64*size_multiplier, 1, name="mlp1_conv2")(x)

    if ARGS.use_tnet_2:
        # feature transform
        x, feat_trans_matrix = pointnet_transform(x, batchsize=ARGS.batchsize, kind="feature") # (B,N,K)
        x = customlayers.ExpandDims(axis=2, name="add_channels_3")(x) # (B,N,1,K)

    local_features = x

    # mlp 2
    x = pointnet_conv(64*size_multiplier, 1, name="mlp2_conv1")(x)
    x = pointnet_conv(128*size_multiplier, 1, name="mlp2_conv2")(x)
    x = pointnet_conv(1024*size_multiplier, 1, name="mlp2_conv3")(x)

    # symmetric function: max pooling
    x = customlayers.ReduceDims(axis=2, name="remove_channels")(x) # (B,N,K)
    x = layers.GlobalMaxPool1D(name="global_feature_maxpool")(x) # (B,K)

    global_feature = x

    # output flow
    if ARGS.output_mode == "seg":
        output = seg_output_flow(local_features, global_feature, output_channels)
    elif ARGS.output_mode == "count":
        output = cls_output_flow(global_feature, output_channels)
    elif ARGS.output_mode == "dense":
        output = dense_output_flow_2(global_feature, ARGS.out_npoints, output_channels)
    else:
        raise NotImplementedError()
    
    # optional post processing for some methods
    if ARGS.output_mode in ("seg", "dense"):
        assert output_channels == 3
        pts = output[...,:2]
        confs = output[...,-1:]
        # limit location coords to 0-1
        pts = layers.Activation("sigmoid")(pts)
        # limit confidence to >= 0
        confs = layers.Activation("softplus")(confs)
        output = layers.Concatenate(axis=-1)([pts, confs])
    if ARGS.loss == "treetop":
        assert output_channels == 1
        # limit to 0 to 1
        output = customlayers.Activation("sigmoid", name="pwtt-sigmoid")(output)
        # add input xy locations to each point
        output = layers.Concatenate(axis=-1, name="pwtt-concat_inpt")([xy_locs, output])

    model = Model(inpt, output)

    if ARGS.use_tnet_2:
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
    from src import patch_generator

    model = pointnet(None, 3, 1)

    traingen, _, _ = patch_generator.get_train_val_gen(ARGS.mode, ARGS.batchsize)

    x, y = traingen[3]

    model.compile()

    out = model.predict(x)

    print(out)



