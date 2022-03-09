import tensorflow as tf
from tensorflow.keras.layers import Layer, BatchNormalization

from src import CUSTOM_LAYERS
from src.pnet2 import utils


class Pointnet_SA(Layer):
    """
    set abstraction
    """

    def __init__(self, npoint, radius, nsample, mlp, group_all=False, knn=False, 
        use_xyz=True, bn=False, **kwargs):

        super(Pointnet_SA, self).__init__(**kwargs)

        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.mlp = mlp
        self.group_all = group_all
        self.knn = False
        self.use_xyz = use_xyz
        self.bn = bn

        self.mlp_list = []

    def build(self, input_shape):
        from src.models import pointnet_conv
        for i, n_filters in enumerate(self.mlp):
            self.mlp_list.append(
                pointnet_conv(n_filters, bn=self.bn, name=self.name+f"_{i}")
            )

        super(Pointnet_SA, self).build(input_shape)

    def call(self, xyz, points, training=True):

        if points is not None:
            if len(points.shape) < 3:
                points = tf.expand_dims(points, axis=0)

        if self.group_all:
            nsample = xyz.get_shape()[1]
            new_xyz, new_points, idx, grouped_xyz = utils.sample_and_group_all(xyz, points, self.use_xyz)
        else:
            new_xyz, new_points, idx, grouped_xyz = utils.sample_and_group(
                self.npoint,
                self.radius,
                self.nsample,
                xyz,
                points,
                self.knn,
                use_xyz=self.use_xyz
            )

        for i, mlp_layer in enumerate(self.mlp_list):
            new_points = mlp_layer(new_points, training=training)

        new_points = tf.math.reduce_max(new_points, axis=2, keepdims=True)

        return new_xyz, tf.squeeze(new_points)

    def compute_output_shape(self, input_shape):
        batchsize = input_shape[0]
        return (batchsize, self.npoints, 1)

    def get_config(self):
        config = super().get_config()
        return {
            "npoint": self.npoint,
            "radius": self.radius,
            "nsample": self.nsample,
            "mlp": self.mlp,
            "group_all": self.group_all,
            "knn": self.knn,
            "use_xyz": self.use_xyz,
            "bn": self.bn,
            **config
        }




class Pointnet_SA_MSG(Layer):
    """
    set abstraction + multi-scale grouping
    """

    def __init__(
        self, npoint, radius_list, nsample_list, mlp, use_xyz=True, bn=False, **kwargs
    ):
        super(Pointnet_SA_MSG, self).__init__(**kwargs)

        self.npoint = npoint
        self.radius_list = radius_list
        self.nsample_list = nsample_list
        self.mlp = mlp
        self.use_xyz = use_xyz
        self.bn = bn

        self.mlp_list = []

    def build(self, input_shape):
        from src.models import pointnet_conv

        for i in range(len(self.radius_list)):
            tmp_list = []
            for j, n_filters in enumerate(self.mlp[i]):
                tmp_list.append(
                    pointnet_conv(n_filters, bn=self.bn, name=self.name+f"_{i}_{j}")
                )
            self.mlp_list.append(tmp_list)

        super(Pointnet_SA_MSG, self).build(input_shape)

    def call(self, xyz, points, training=True):

        if points is not None:
            if len(points.shape) < 3:
                points = tf.expand_dims(points, axis=0)

        new_xyz = utils.gather_point(xyz, utils.farthest_point_sample(self.npoint, xyz))

        new_points_list = []

        for i in range(len(self.radius_list)):
            radius = self.radius_list[i]
            nsample = self.nsample_list[i]
            idx, pts_cnt = utils.query_ball_point(radius, nsample, xyz, new_xyz)
            grouped_xyz = utils.group_point(xyz, idx)
            grouped_xyz -= tf.tile(tf.expand_dims(new_xyz, 2), [1,1,nsample,1])

            if points is not None:
                grouped_points = utils.group_point(points, idx)
                if self.use_xyz:
                    grouped_points = tf.concat([grouped_points, grouped_xyz], axis=-1)
            else:
                grouped_points = grouped_xyz

            for i, mlp_layer in enumerate(self.mlp_list[i]):
                grouped_points = mlp_layer(grouped_points, training=training)

            new_points = tf.math.reduce_max(grouped_points, axis=2)
            new_points_list.append(new_points)

        new_points_concat = tf.concat(new_points_list, axis=-1)

        return new_xyz, new_points_concat

    def get_config(self):
        config = super().get_config()
        return {
            "npoint": self.npoint,
            "radius_list": self.radius_list,
            "nsample_list": self.nsample_list,
            "mlp": self.mlp,
            "use_xyz": self.use_xyz,
            "bn": self.bn,
            **config
        }



class Pointnet_FP(Layer):
    """
    feature propogation
    """

    def __init__(
        self, mlp, bn=False, **kwargs
    ):

        super(Pointnet_FP, self).__init__(**kwargs)

        self.mlp = mlp
        self.bn = bn

        self.mlp_list = []


    def build(self, input_shape):
        from src.models import pointnet_conv
        for i, n_filters in enumerate(self.mlp):
            self.mlp_list.append(
                pointnet_conv(n_filters, bn=self.bn, name=self.name+f"_{i}")
            )
        super(Pointnet_FP, self).build(input_shape)

    def call(self, xyz1, xyz2, points1, points2, training=True):

        if points1 is not None:
            if len(points1.shape) < 3:
                points1 = tf.expand_dims(points1, axis=0)
        if points2 is not None:
            if len(points2.shape) < 3:
                points2 = tf.expand_dims(points2, axis=0)

        dist, idx = utils.three_nn(xyz1, xyz2)
        dist = tf.maximum(dist, 1e-10)
        norm = tf.reduce_sum((1.0/dist),axis=2, keepdims=True)
        norm = tf.tile(norm,[1,1,3])
        weight = (1.0/dist) / norm
        interpolated_points = utils.three_interpolate(points2, idx, weight)

        if points1 is not None:
            new_points1 = tf.concat(axis=2, values=[interpolated_points, points1]) # B,ndataset1,nchannel1+nchannel2
        else:
            new_points1 = interpolated_points
        new_points1 = tf.expand_dims(new_points1, 2)

        for i, mlp_layer in enumerate(self.mlp_list):
            new_points1 = mlp_layer(new_points1, training=training)

        new_points1 = tf.squeeze(new_points1)
        if len(new_points1.shape) < 3:
            new_points1 = tf.expand_dims(new_points1, axis=0)

        return new_points1


    def get_config(self):
        config = super().get_config()
        return {
            "mlp": self.mlp,
            "bn": self.bn,
            **config
        }


CUSTOM_LAYERS["Pointnet_FP"] = Pointnet_FP
CUSTOM_LAYERS["Pointnet_SA"] = Pointnet_SA
CUSTOM_LAYERS["Pointnet_SA_MSG"] = Pointnet_SA_MSG
