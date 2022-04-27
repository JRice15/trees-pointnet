import argparse
import os, sys

import open3d
import numpy as np

from src import DATA_DIR, ARGS
from src.utils import get_default_dsname, get_all_regions
from src.patch_generator import get_train_val_gens, get_test_gen, get_tvt_split


def viz_pointcloud(xyz):
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(xyz)

    bbox = open3d.geometry.AxisAlignedBoundingBox.create_from_points(pcd.points)
    print(bbox)

    viewer = open3d.visualization.Visualizer()
    viewer.create_window()
    viewer.add_geometry(pcd)

    # opt = viewer.get_render_option()
    #opt.show_coordinate_frame = True
    #opt.background_color = np.asarray([0.5, 0.5, 0.5])

    viewer.run()
    viewer.destroy_window()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dsname",help="dataset name")
    parser.add_argument("--region","-r",required=True)
    parser.add_argument("--patch","-p","-n",type=int,required=True,help="patch number")
    parser.parse_args(namespace=ARGS)

    if ARGS.dsname is None:
        ARGS.dsname = get_default_dsname()

    # from raw data

    path = DATA_DIR.joinpath("lidar", ARGS.dsname, "regions", ARGS.region, "lidar_patch_{}.npy".format(ARGS.patch))
    xyz = np.load(path.as_posix())[:,:3]
    xyz = xyz[xyz[:,2] <= 50]

    viz_pointcloud(xyz)


    # from training dataset loader

    # ARGS.batchsize = 1
    # ARGS.subdivide = 4
    # ARGS.loss = "mmd"
    # ARGS.npoints = 800
    # ARGS.noise_sigma = 0
    # ARGS.test = False

    # all_regions = get_all_regions(ARGS.dsname)

    # patch_id = (ARGS.region, int(ARGS.patch))
    # train_ids, val_ids, test_ids = get_tvt_split(ARGS.dsname, all_regions)
    # if patch_id in train_ids:
    #     ds, _ = get_train_val_gens(ARGS.dsname, all_regions)
    # elif patch_id in val_ids:
    #     _, ds = get_train_val_gens(ARGS.dsname, all_regions)
    # elif patch_id in test_ids:
    #     ds = get_test_gen(ARGS.dsname, all_regions)
    # else:
    #     raise ValueError("Bad patch: {}".format(patch_id))
    
    # pts, gt, _ = ds.get_patch(ARGS.region, ARGS.patch, 0)

    # viz_pointcloud(pts[:,:3])

    
if __name__ == "__main__":
    main()