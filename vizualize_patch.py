import argparse
import os, sys

import open3d
import numpy as np

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
    from src import DATA_DIR, ARGS
    from src.utils import get_default_dsname, get_all_regions
    from src.patch_generator import get_train_val_gens, get_test_gen, get_tvt_split

    parser = argparse.ArgumentParser()
    parser.add_argument("--dsname",help="dataset name")
    parser.add_argument("--region","-r",required=True)
    parser.add_argument("--patch","-p","-n",type=int,required=True,help="patch number")
    parser.add_argument("--minz",type=int,default=-10)
    parser.add_argument("--maxz",type=int,default=50)
    args = parser.parse_args()

    if args.dsname is None:
        args.dsname = get_default_dsname()

    # from raw data


    path = DATA_DIR.joinpath("lidar", args.dsname, "regions", args.region, "lidar_patch_{}.npy".format(args.patch))
    xyz = np.load(path.as_posix())[:,:3]
    xyz = xyz[(xyz[:,2] <= args.maxz) & (args.minz <= xyz[:,2])]

    print(xyz.shape)

    viz_pointcloud(xyz)


    # from training dataset loader

    # set fake dataset params
    ARGS.dsname = args.dsname
    ARGS.handle_small = "fill"
    ARGS.batchsize = 1
    ARGS.subdivide = 5
    ARGS.loss = "mmd"
    ARGS.npoints = 1995
    ARGS.noise_sigma = 0
    ARGS.test = False

    # all_regions = get_all_regions(args.dsname)
    regions = [args.region]

    patch_id = (args.region, int(args.patch))
    train_ids, val_ids, test_ids = get_tvt_split(args.dsname, regions)
    if patch_id in train_ids:
        ds, _ = get_train_val_gens(args.dsname, regions)
    elif patch_id in val_ids:
        _, ds = get_train_val_gens(args.dsname, regions)
    elif patch_id in test_ids:
        ds = get_test_gen(args.dsname, regions)
    else:
        raise ValueError("Bad patch: {}".format(patch_id))
    
    print(ds.summary())

    # print(sorted(ds.patch_ids))

    i = 0
    all_xyz = None
    while True:
        subpatch_id = (args.region, args.patch, i)
        try:
            pts, _, _ = ds.get_patch(*subpatch_id)
        except ValueError as e:
            print(e)
            break

        pts = ds.denormalize_pts(pts, subpatch_id)
        xyz = pts[:,:3]
        xyz = xyz[(xyz[:,2] <= args.maxz) & (args.minz <= xyz[:,2])]

        if all_xyz is None:
            all_xyz = xyz
        else:
            all_xyz = np.concatenate([all_xyz, xyz], axis=0)
        
        i += 1

    viz_pointcloud(all_xyz)

    
if __name__ == "__main__":
    main()