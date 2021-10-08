import os
import re
import sys
import glob
from pathlib import PurePath
import argparse

import geopandas as gpd
import h5py
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import tables

matplotlib.rc_file_defaults()

sys.path.append("..")

from core.utils import raster_plot


parser = argparse.ArgumentParser()
parser.add_argument("--dsname",required=True,help="name of the generated dataset")
ARGS = parser.parse_args()


regions = glob.glob("../data/generated/"+ARGS.dsname+"/*")

for region_file in regions:
    region_file = PurePath(region_file)
    region_name = region_file.stem
    print(region_name, region_file)

    os.makedirs("output/{}/example_patches".format(region_name), exist_ok=True)
    outdir = PurePath("output/{}".format(region_name))

    with h5py.File(region_file.as_posix(), "r") as hf:
        grid = hf["/grid"][:]
    
    grid_x = grid[:,0]
    grid_y = grid[:,1]

    plt.scatter(grid_x, grid_y)
    plt.title("Patch locations")
    plt.savefig(outdir.joinpath("patch_locations").as_posix)
    plt.show()

    exit()

    fig, axlist = plt.subplots(2,2,figsize=(12,9))
    for i,fname in enumerate(("train", "test")):
        for j,kind in enumerate(("lidar", "gt")):
            ax = axlist[i][j]
            plt.sca(ax)

            full_lens = []
            with tables.open_file("../data/{}_patches.h5".format(fname), "r") as train_fp:
                for y in range(ROWS):
                    full_lens.append([])
                    for x in range(COLS):
                        try:
                            num = train_fp.get_node("/{}/patch{}_{}".format(kind,x,y)).shape[0]
                        except:
                            num = 0
                        full_lens[-1].append(num)

            x, y = np.meshgrid(grid_x, grid_y, indexing='xy')
            plt.pcolormesh(x, y, full_lens, shading='auto')
            plt.colorbar()
            plt.title("{} {}".format(fname, kind))

    plt.savefig("output/patches_raster.png", dpi=200)
    plt.close()

    with tables.open_file("../data/train_patches.h5", "r") as train_fp, \
            tables.open_file("../data/test_patches.h5", "r") as test_fp:

        train_attrs = train_fp.get_node("/lidar")._v_attrs
        test_attrs = test_fp.get_node("/lidar")._v_attrs
        for i in train_attrs._f_list():
            print("train", i, train_attrs[i])
        for i in test_attrs._f_list():
            print("test", i, test_attrs[i])
        train_attrs = train_fp.get_node("/gt")._v_attrs
        test_attrs = test_fp.get_node("/gt")._v_attrs
        for i in train_attrs._f_list():
            print("train", i, train_attrs[i])
        for i in test_attrs._f_list():
            print("test", i, test_attrs[i])

        train_patch_lens = [i.shape[0] for i in train_fp.get_node("/lidar")]
        train_gt_lens = [i.shape[0] for i in train_fp.get_node("/gt")]
        test_patch_lens = [i.shape[0] for i in test_fp.get_node("/lidar")]
        test_gt_lens = [i.shape[0] for i in test_fp.get_node("/gt")]

        pts_min = min(min(train_patch_lens), min(test_patch_lens))
        pts_max = max(max(train_patch_lens), max(test_patch_lens))
        step = (pts_max - pts_min) // 20
        plt.hist([train_patch_lens, test_patch_lens], label=["train", "test"], density=True, bins=range(pts_min, pts_max+1, step))
        plt.legend()
        plt.savefig("output/pts_per_patch")
        plt.close()

        trees_min = min(min(train_gt_lens), min(test_gt_lens))
        trees_max = max(max(train_gt_lens), max(test_gt_lens))
        step = (trees_max - trees_min) // 20
        plt.hist([train_gt_lens, test_gt_lens], label=["train", "test"], density=True, bins=range(trees_min, trees_max+1, step))
        plt.legend()
        plt.savefig("output/trees_per_patch")
        plt.close()

        train_heights = [i[:,2].max() for i in train_fp.get_node("/lidar")]
        test_heights = [i[:,2].max() for i in train_fp.get_node("/lidar")]
        plt.hist([train_heights, test_heights], label=["train", "test"], density=True)
        plt.legend()
        plt.savefig("output/patch_max_heights")
        plt.close()


    with h5py.File("../data/test_patches.h5", "r") as f:
        print("generating visualizations")
        # vizualize 20 patches
        keys = sorted(list(f["lidar"].keys()))
        for i in range(0, len(keys), len(keys)//20):
            patchname = keys[i]
            x = f["lidar/"+patchname][:]
            y = f["gt/"+patchname][:]
            naip = f["naip/"+patchname][:]
            
            xlocs = x[...,:2]
            xweights = x[...,2]
            raster_plot(xlocs, weights=xweights, mode="max", gaussian_sigma=0.04, mark=y,
                filename="output/example_patches/{}_lidar_height.png".format(patchname))

            x_ndvi = x[...,3]
            raster_plot(xlocs, weights=x_ndvi, mode="max", gaussian_sigma=0.04, mark=y,
                filename="output/example_patches/{}_lidar_ndvi.png".format(patchname))

            step = len(x) // 3000
            if step > 0:
                leftover = len(x) % 3000
                x_sampled = x[0:len(x)-leftover:step]
                assert len(x_sampled) == 3000
                xlocs = x_sampled[...,:2]
                xweights = x_sampled[...,2]
                raster_plot(xlocs, weights=xweights, mode="max", gaussian_sigma=0.04, mark=y,
                    filename="output/example_patches/{}_lidar_height_sampled3k.png".format(patchname))

            plt.imshow(naip[...,:3])
            plt.colorbar()
            plt.tight_layout()
            plt.savefig("output/example_patches/"+patchname+"_NAIP.png")
            plt.clf()
            plt.close()


        xys = [f["gt/"+key][:] for key in keys]
        xys = np.concatenate(xys, axis=0)
        xs, ys = xys[:,0], xys[:,1]

        trees = gpd.GeoDataFrame(
            geometry=gpd.points_from_xy(xs, ys),
            crs="EPSG:26911")
        
        trees.to_file("output/test_patches.shp")



    with h5py.File("../data/train_patches.h5", "r") as f:
        assert len(f["lidar"].keys()) == len(f["gt"].keys())
        print("First 10 Lidar, GT patches:")
        print(list(f["lidar"].keys())[:10])
        print(list(f["gt"].keys())[:10])


# if ROWS == 44:
#     """
#     verify correct patches are selected for test
#     """
#     with h5py.File("../data/test_patches.h5", "r") as f:
#         keys = list(f["gt"].keys())
#         keys = [re.sub(r"patch", "", i) for i in keys]
#         keys = [i.split("_") for i in keys]
#         keys = [(int(x), int(y)) for x,y in keys]
#         keys.sort()

#     # a few of the test patches, hand indexed. These are indexed from 1, like the grid
#     expected = [
#         (25,7),
#         (29,8),
#         (32,10),
#         (21,10),
#         (25,14)
#     ]

#     expected.sort()
#     assert all([i in keys for i in expected])
