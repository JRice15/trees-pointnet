import argparse
import os, sys

import open3d

import numpy as np

from common import DATA_DIR
from common.data_handling import get_default_dsname, get_all_regions


def viz_pointcloud(xyz, point_size=6, colors=None):
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(xyz)
    if colors is not None:
        pcd.colors = open3d.utility.Vector3dVector(colors[:, :3])

    bbox = open3d.geometry.AxisAlignedBoundingBox.create_from_points(pcd.points)
    print(bbox)

    viewer = open3d.visualization.Visualizer()
    viewer.create_window()
    viewer.add_geometry(pcd)

    opt = viewer.get_render_option()
    opt.point_size = point_size

    viewer.run()
    viewer.destroy_window()



def viz_from_ds(args):
    from pointnet.src import ARGS
    from pointnet.src.patch_generator import LidarPatchGen

    if args.name is None:
        args.name = get_default_dsname()

    ### visualize pts from training dataset loader

    # set fake dataset params
    ARGS.dsname = args.name
    ARGS.handle_small = "drop"
    ARGS.batchsize = 1
    ARGS.subdivide = 5
    ARGS.loss = "mmd"
    ARGS.npoints = 1995
    ARGS.noise_sigma = 0
    ARGS.test = False

    region, patch = args.pid
    patch_id = (region, int(patch))

    regions = [region]
    ds = LidarPatchGen([patch_id], name="viz", batchsize=1, training=False)

    print(ds.summary())
    # print(sorted(ds.patch_ids))

    all_xyz = None
    for i in range(ARGS.subdivide*2-1):
        for j in range(ARGS.subdivide*2-1):
            subpatch_id = patch_id + (i, j)
            try:
                pts, _, _ = ds.get_patch(*subpatch_id)
            except ValueError as e:
                print(e)
                continue

            pts = ds.denormalize_pts(pts, subpatch_id)
            xyz = pts[:,:3]
            xyz = xyz[(xyz[:,2] <= args.maxz) & (args.minz <= xyz[:,2])]

            if all_xyz is None:
                all_xyz = xyz
            else:
                all_xyz = np.concatenate([all_xyz, xyz], axis=0)

    viz_pointcloud(all_xyz)


def viz_from_raw_ds(args):
    if args.name is None:
        args.name = get_default_dsname()

    ### visualize pts from raw data
    region, patch = args.pid

    path = DATA_DIR.joinpath("lidar", args.name, "regions", region, "lidar_patch_{}.npy".format(patch))
    xyz = np.load(path.as_posix())[:,:3]
    xyz = xyz[(xyz[:,2] <= args.maxz) & (args.minz <= xyz[:,2])]

    print(xyz.shape)

    viz_pointcloud(xyz)


def viz_from_preds(args):
    from pointnet.src.utils import glob_modeldir

    path = glob_modeldir(args.name).joinpath("results_test", "raw_preds.npz")
    npz = np.load(path.as_posix())

    xyz = npz["_".join(args.pid)]
    xyz = xyz[(xyz[:,2] <= args.maxz) & (args.minz <= xyz[:,2])]

    # scale for better visibility
    xyz[:,2] = xyz[:,2] * 10

    viz_pointcloud(xyz)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source",choices=("raw", "ds", "preds"),default="raw",help="source of pts to load from")
    parser.add_argument("--name",help="name of particular source; dsname for `raw` and `ds`, or model name for `preds`")
    parser.add_argument("--pid","-p",nargs=2,required=True)
    parser.add_argument("--minz",type=float,default=-10)
    parser.add_argument("--maxz",type=float,default=50)
    args = parser.parse_args()

    if args.source == "raw":
        viz_from_raw_ds(args)
    elif args.source == "ds":
        viz_from_ds(args)
    elif args.source == "preds":
        viz_from_preds(args)
    else:
        raise ValueError()


    
if __name__ == "__main__":
    main()