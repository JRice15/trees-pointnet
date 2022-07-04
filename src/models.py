import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras import backend as K
from tensorflow.keras import layers
from tensorflow import keras

from src import customlayers, ARGS


def pointnet_conv(outchannels, name, bn=True, activation=True):
    """
    returns callable, which creates pipeline of conv, batchnorm, and relu
    input: (B,N,1,K)
    args:
        activation: bool (defaults to relu if true), or activation name otherwise
    """
    layer_list = [
        layers.Conv2D(int(outchannels), kernel_size=1, padding="valid",
                    strides=1, kernel_initializer="glorot_normal", name=name+"_conv")
    ]
    if bn:
        layer_list.append(layers.BatchNormalization(name=name+"_bn"))
    if activation:
        if activation is True:
            layer_list.append(layers.ReLU(name=name+"_relu"))
        else:
            layer_list.append(layers.Activation(activation, name=name+"_act"))

    return keras.Sequential(layer_list, name=name)


def pointnet_dense(outchannels, name, bn=True, activation=True):
    """
    returns callable, which creates pipeline of dense, batchnorm, and relu
    input: (B,N,K)
    """
    seq = keras.Sequential([
        layers.Dense(int(outchannels), kernel_initializer="glorot_normal", name=name+"_dense")
    ], name=name)
    if bn:
        seq.add( layers.BatchNormalization(name=name+"_bn") )
    if activation:
        seq.add( layers.ReLU(name=name+"_relu") )
    return seq


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
    nattributes = x_in.shape[-1]

    x = x_in

    x = pointnet_conv(64, name=prefix+"conv1")(x)
    x = pointnet_conv(128, name=prefix+"conv2")(x)
    x = pointnet_conv(1024, name=prefix+"conv3")(x)

    x = customlayers.ReduceDims(axis=2, name=prefix+"squeeze")(x) # (B,N,K)
    x = layers.GlobalMaxPool1D(name=prefix+"maxpool")(x) # (B,K)

    x = pointnet_dense(512, name=prefix+"dense1")(x)
    x = pointnet_dense(256, name=prefix+"dense2")(x)

    trans_matrix = customlayers.TNet(nattributes, name=prefix+"tnet")(x) # (B,(K*K))

    x_in = customlayers.ReduceDims(axis=2, name=prefix+"squeeze2")(x_in)
    
    # apply transformation matrix
    x_out = customlayers.MatMul(name=prefix+"matmul")([x_in, trans_matrix])

    return x_out, trans_matrix


def seg_output_flow(local_features, global_feature, size_multiplier, outchannels):
    """
    Base segmentation output flow
    Returns:
        per-point feature vectors: (B,N,outchannels)
    example shape: (16, 500, 3)
    """
    # concat local and global
    global_len = global_feature.shape[-1]
    global_feature = layers.Reshape((1,1,global_len), name="expand_global_feature_reshape")(global_feature)
    _, npoints, _, _ = local_features.shape
    global_feature = customlayers.Tile(npoints, axis=1)(global_feature)

    full_features = layers.Concatenate(axis=-1, name="seg_concat")([local_features, global_feature])

    # output flow
    x = full_features
    x = pointnet_conv(512*size_multiplier, name="outmlp_conv1")(x)
    if ARGS.dropout_rate > 0:
        x = layers.Dropout(ARGS.dropout_rate, name="outmlp_dropout1")(x)

    x = pointnet_conv(256*size_multiplier, name="outmlp_conv2")(x)
    if ARGS.dropout_rate > 0:
        x = layers.Dropout(ARGS.dropout_rate, name="outmlp_dropout2")(x)

    x = pointnet_conv(128*size_multiplier, name="outmlp_conv3")(x)
    if ARGS.dropout_rate > 0:
        x = layers.Dropout(ARGS.dropout_rate, name="outmlp_dropout3")(x)

    last_size = min(128, 128*size_multiplier)
    x = pointnet_conv(last_size, name="outmlp_conv4")(x)
    if ARGS.dropout_rate > 0:
        x = layers.Dropout(ARGS.dropout_rate, name="outmlp_dropout4")(x)

    x = pointnet_conv(outchannels, name="outmlp_conv_final",
                      bn=False, activation=False)(x)
    x = customlayers.ReduceDims(axis=2, name="outmlp_squeeze")(x)
    
    return x

def cls_output_flow(global_feature, outchannels):
    """
    Base global classification output flow
    returns:
        global classification vector: (B,outchannels)
    example shape: (16, 3)
    """
    x = global_feature

    x = pointnet_dense(512, "outmlp_dense1")(x)
    if ARGS.dropout_rate > 0:
        x = layers.Dropout(ARGS.dropout_rate, name="outmlp_dropout1")(x)
    x = pointnet_dense(256, "outmlp_dense2")(x)
    if ARGS.dropout_rate > 0:
        x = layers.Dropout(ARGS.dropout_rate, name="outmlp_dropout2")(x)

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
    returns:
        features for user-specified number of points: (B, outpoints, outchannels)
    example shape: (16, 256, 3)
    """
    x = global_feature

    input_size = x.shape[-1]
    target_size = out_npoints * out_channels
    intermediate_size = (input_size + target_size) // 2

    x = pointnet_dense(intermediate_size, "outmlp_dense1")(x)
    if ARGS.dropout_rate > 0:
        x = layers.Dropout(ARGS.dropout_rate, name="outmlp_dropout1")(x)
    x = pointnet_dense(target_size, "outmlp_dense2")(x)
    
    x = layers.Reshape((out_npoints, out_channels), name="outmlp_reshape")(x)
    if ARGS.dropout_rate > 0:
        x = layers.Dropout(ARGS.dropout_rate, name="outmlp_dropout2")(x)

    x = pointnet_dense(out_channels, bn=False, activation=False, name="out_channels_dense")(x)

    return x




def pointnet_1(x, size_multiplier):
    """
    original pointnet
    args:
        x: input tensor, shape (B,N,K)
    returns:
        local features tensor (B,N,1,K)
        global feature tensor (B,K)
        dict mapping loss names to losses
    """
    output_losses = {}
    
    x = customlayers.ExpandDims(axis=2, name="add_channels_1")(x) # (B,N,1,K)

    if ARGS.use_tnet_1:
        # input transform
        x, inpt_trans_matrix = pointnet_transform(x, batchsize=ARGS.batchsize, kind="input") # (B,N,K)
        x = customlayers.ExpandDims(axis=2, name="add_channels_2")(x) # (B,N,1,K)

    # mlp 1
    x = pointnet_conv(64*size_multiplier, name="mlp1_conv1")(x)
    x = pointnet_conv(64*size_multiplier, name="mlp1_conv2")(x)

    if ARGS.use_tnet_2:
        # feature transform
        x, feat_trans_matrix = pointnet_transform(x, batchsize=ARGS.batchsize, kind="feature") # (B,N,K)
        x = customlayers.ExpandDims(axis=2, name="add_channels_3")(x) # (B,N,1,K)

    local_features = x

    # mlp 2
    x = pointnet_conv(64*size_multiplier, name="mlp2_conv1")(x)
    x = pointnet_conv(128*size_multiplier, name="mlp2_conv2")(x)
    x = pointnet_conv(1024*size_multiplier, name="mlp2_conv3")(x)

    # symmetric function: max pooling
    x = customlayers.ReduceDims(axis=2, name="remove_channels")(x) # (B,N,K)
    x = layers.GlobalMaxPool1D(name="global_feature_maxpool")(x) # (B,K)

    global_feature = x
    
    if ARGS.use_tnet_2:
        # feature transformation matrix orthogonality loss
        dims = feat_trans_matrix.shape[1]
        ortho_diff = tf.matmul(feat_trans_matrix,
                        tf.transpose(feat_trans_matrix, perm=[0,2,1]))
        ortho_diff -= tf.constant(tf.eye(dims), dtype=K.floatx())
        ortho_loss = ARGS.ortho_weight * tf.nn.l2_loss(ortho_diff)
        output_losses["ortho_loss"] = ortho_loss

    return local_features, global_feature, output_losses



def pointnet_2(inputs, npoints, size_multiplier):
    """
    adapted from:
    https://github.com/dgriffiths3/pointnet2-tensorflow2

    args:
        inputs: input tensor (B,N,channels)
    returns:
        local features tensor (B,N,1,channels)
        global feature tensor (B,channels)
        (always empty) loss dict
    """
    from src.pnet2.layers import Pointnet_SA_MSG, Pointnet_SA

    if inputs.get_shape()[-1] > 3:
        xyz = inputs[...,:3]
        input_features = inputs[...,3:]
    else:
        xyz = inputs
        input_features = None

    # TODO size multiplier

    # base sizes
    sample_sizes = np.array([16, 32, 64]) * ARGS.size_multiplier
    mlp_sizes = np.array([
        [32, 32, 64], [64, 64, 128], [64, 96, 128]
    ]) * ARGS.size_multiplier


    layer1 = Pointnet_SA_MSG(
        npoint=npoints,
        radius_list=[0.1, 0.2, 0.4],
        nsample_list=sample_sizes,
        mlp=mlp_sizes,
        bn=ARGS.use_batchnorm,
    )
    xyz, local_features = layer1(xyz, input_features)

    layer2 = Pointnet_SA_MSG(
        npoint=512 * ARGS.size_multiplier,
        radius_list=[0.2,0.4,0.8],
        nsample_list=2*sample_sizes,
        mlp=2*mlp_sizes,
        bn=ARGS.use_batchnorm
    )
    xyz, intermediate_features = layer2(xyz, local_features)

    layer3 = Pointnet_SA(
        npoint=None,
        radius=None,
        nsample=None,
        mlp=np.array([256, 512, 1024]) * ARGS.size_multiplier,
        group_all=True,
        bn=ARGS.use_batchnorm,
    )
    xyz, global_feature = layer3(xyz, intermediate_features)

    local_features = customlayers.ExpandDims(axis=2)(local_features) # (B,N,1,K)

    return local_features, global_feature, {}



def pointnet(inpt_shape, size_multiplier, output_channels):
    """
    args:
        inpt_shape: tuple
        size_mult: multiply channels by this amount
        output_channels: num features per output point
    """
    npoints, nattributes = inpt_shape
    inpt = layers.Input(inpt_shape, name="inpt_layer", 
        # batch_size=ARGS.batchsize
    ) # (B,N,K)

    xy_locs = inpt[...,:2]

    x = inpt

    if ARGS.use_pnet2:
        local_features, global_feature, losses = pointnet_2(x, npoints, size_multiplier)
    else:
        local_features, global_feature, losses = pointnet_1(x, size_multiplier)

    # output flow
    if ARGS.output_mode == "seg":
        output = seg_output_flow(local_features, global_feature, size_multiplier, output_channels)
    elif ARGS.output_mode == "dense":
        output = dense_output_flow_2(global_feature, ARGS.out_npoints, output_channels)
    else:
        raise NotImplementedError("unknown output flow")

    # post processing
    assert output_channels == 3
    pts = output[...,:2]
    confs = output[...,2:]
    # limit location coords to 0-1
    pts = layers.ReLU(max_value=1.0, name="loc-lim")(pts)
    # limit confidence to >= 0
    confs = layers.Activation(ARGS.conf_act, name="conf-lim")(confs)
    output = layers.Concatenate(axis=-1, name="final-concat")([pts, confs])

    model = Model(inpt, output)

    for loss_name, loss_val in losses.items():
        model.add_loss(loss_val)
        model.add_metric(loss_val, name=loss_name, aggregation="mean")

    return model




