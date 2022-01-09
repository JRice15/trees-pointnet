"""
output plots and other stats data about a dataset into the

"""

import os
import re
import sys
import json
import glob
from pathlib import PurePath
import argparse

import geopandas as gpd
import h5py
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.cm as cm
import numpy as np
import pandas as pd
import tables
import seaborn

# add parent directory
dn = os.path.dirname
sys.path.append(dn(dn(os.path.abspath(__file__))))

from core import DATA_DIR
from core.utils import raster_plot



def jitter(vals, percent):
    """
    vals: values to jitter
    percent: percent (0-100) of the total range (from max(vals) to min(vals)) to jitter by
    """
    range_ = max(vals) - min(vals)
    jittersize = range_ * percent * 0.01
    # generate 0 to 1
    rand = np.random.random_sample(len(vals))
    # convert to -1 to 1
    rand = (rand * 2) - 1
    return vals + (rand * jittersize)


def main(dsname):
    print("Running analyze_dataset.py")
    combined_stats = None

    DS_DIR = DATA_DIR.joinpath("generated", ARGS.dsname)
    region_files = glob.glob(DS_DIR.joinpath("*.h5").as_posix())

    for region_file in region_files:
        region_file = PurePath(region_file)
        region_name = region_file.stem
        print("Analyzing", region_name, region_file)

        outdir = DS_DIR.joinpath("stats", region_name)
        os.makedirs(outdir.as_posix(), exist_ok=True)

        stats = pd.Series(dtype="int")
        with h5py.File(region_file.as_posix(), "r") as hf:
            stats["n_patches"] = len(hf["lidar"].keys())
            stats["total_lidar_points"] = sum([pts.shape[0] for pts in hf["lidar"].values()])
            stats["total_gt_trees"] = sum([pts.shape[0] for pts in hf["gt"].values()])
            patch_coords = hf["/grid"][:]

        grid_x = patch_coords[...,0]
        grid_y = patch_coords[...,1]

        patch_df = pd.DataFrame({"x": grid_x, "y": grid_y})
        patch_df["x_jit"] = jitter(patch_df["x"], 3)
        patch_df["y_jit"] = jitter(patch_df["y"], 3)

        with h5py.File(region_file.as_posix(), "r") as hf:
            patch_ids = hf["lidar"].keys()
            n_patches = len(patch_ids)
            patch_df["pt_count"] = [hf["lidar/patch"+str(pnum)].shape[0] for pnum in range(n_patches)]
            patch_df["gt_count"] = [hf["gt/patch"+str(pnum)].shape[0] for pnum in range(n_patches)]

            seaborn.scatterplot(data=patch_df, x="x_jit", y="y_jit", hue="pt_count")
            plt.title("Patch locations")
            plt.savefig(outdir.joinpath("patch_locations").as_posix())
            plt.clf()

            seaborn.histplot(data=patch_df, x="pt_count")
            plt.title("Points per patch")
            plt.savefig(outdir.joinpath("pount_counts").as_posix())
            plt.clf()

            seaborn.histplot(data=patch_df, x="gt_count")
            plt.title("Groundtruth trees per patch")
            plt.savefig(outdir.joinpath("gt_counts").as_posix())
            plt.clf()

            # TODO heights, NDVI values
            # heights = 
            # example plots

        with open(outdir.joinpath("stats.json"), "w") as f:
            json.dump(stats.to_dict(), f, indent=2)

        if combined_stats is None:
            combined_stats = stats
        else:
            combined_stats += stats

        # exit()

        # with tables.open_file("../data/train_patches.h5", "r") as train_fp, \
        #         tables.open_file("../data/test_patches.h5", "r") as test_fp:

        #     train_attrs = train_fp.get_node("/lidar")._v_attrs
        #     test_attrs = test_fp.get_node("/lidar")._v_attrs
        #     for i in train_attrs._f_list():
        #         print("train", i, train_attrs[i])
        #     for i in test_attrs._f_list():
        #         print("test", i, test_attrs[i])
        #     train_attrs = train_fp.get_node("/gt")._v_attrs
        #     test_attrs = test_fp.get_node("/gt")._v_attrs
        #     for i in train_attrs._f_list():
        #         print("train", i, train_attrs[i])
        #     for i in test_attrs._f_list():
        #         print("test", i, test_attrs[i])

        #     train_patch_lens = [i.shape[0] for i in train_fp.get_node("/lidar")]
        #     train_gt_lens = [i.shape[0] for i in train_fp.get_node("/gt")]
        #     test_patch_lens = [i.shape[0] for i in test_fp.get_node("/lidar")]
        #     test_gt_lens = [i.shape[0] for i in test_fp.get_node("/gt")]

        #     pts_min = min(min(train_patch_lens), min(test_patch_lens))
        #     pts_max = max(max(train_patch_lens), max(test_patch_lens))
        #     step = (pts_max - pts_min) // 20
        #     plt.hist([train_patch_lens, test_patch_lens], label=["train", "test"], density=True, bins=range(pts_min, pts_max+1, step))
        #     plt.legend()
        #     plt.savefig("output/pts_per_patch")
        #     plt.close()

        #     trees_min = min(min(train_gt_lens), min(test_gt_lens))
        #     trees_max = max(max(train_gt_lens), max(test_gt_lens))
        #     step = (trees_max - trees_min) // 20
        #     plt.hist([train_gt_lens, test_gt_lens], label=["train", "test"], density=True, bins=range(trees_min, trees_max+1, step))
        #     plt.legend()
        #     plt.savefig("output/trees_per_patch")
        #     plt.close()

        #     train_heights = [i[:,2].max() for i in train_fp.get_node("/lidar")]
        #     test_heights = [i[:,2].max() for i in train_fp.get_node("/lidar")]
        #     plt.hist([train_heights, test_heights], label=["train", "test"], density=True)
        #     plt.legend()
        #     plt.savefig("output/patch_max_heights")
        #     plt.close()


        # with h5py.File("../data/test_patches.h5", "r") as f:
        #     print("generating visualizations")
        #     # vizualize 20 patches
        #     keys = sorted(list(f["lidar"].keys()))
        #     for i in range(0, len(keys), len(keys)//20):
        #         patchname = keys[i]
        #         x = f["lidar/"+patchname][:]
        #         y = f["gt/"+patchname][:]
        #         naip = f["naip/"+patchname][:]
                
        #         xlocs = x[...,:2]
        #         xweights = x[...,2]
        #         raster_plot(xlocs, weights=xweights, mode="max", gaussian_sigma=0.04, mark=y,
        #             filename="output/example_patches/{}_lidar_height.png".format(patchname))

        #         x_ndvi = x[...,3]
        #         raster_plot(xlocs, weights=x_ndvi, mode="max", gaussian_sigma=0.04, mark=y,
        #             filename="output/example_patches/{}_lidar_ndvi.png".format(patchname))

        #         step = len(x) // 3000
        #         if step > 0:
        #             leftover = len(x) % 3000
        #             x_sampled = x[0:len(x)-leftover:step]
        #             assert len(x_sampled) == 3000
        #             xlocs = x_sampled[...,:2]
        #             xweights = x_sampled[...,2]
        #             raster_plot(xlocs, weights=xweights, mode="max", gaussian_sigma=0.04, mark=y,
        #                 filename="output/example_patches/{}_lidar_height_sampled3k.png".format(patchname))

        #         plt.imshow(naip[...,:3])
        #         plt.colorbar()
        #         plt.tight_layout()
        #         plt.savefig("output/example_patches/"+patchname+"_NAIP.png")
        #         plt.clf()
        #         plt.close()


        #     xys = [f["gt/"+key][:] for key in keys]
        #     xys = np.concatenate(xys, axis=0)
        #     xs, ys = xys[:,0], xys[:,1]

        #     trees = gpd.GeoDataFrame(
        #         geometry=gpd.points_from_xy(xs, ys),
        #         crs="EPSG:26911")
            
        #     trees.to_file("output/test_patches.shp")



        # with h5py.File("../data/train_patches.h5", "r") as f:
        #     assert len(f["lidar"].keys()) == len(f["gt"].keys())
        #     print("First 10 Lidar, GT patches:")
        #     print(list(f["lidar"].keys())[:10])
        #     print(list(f["gt"].keys())[:10])


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

    with open(DS_DIR.joinpath("stats", "combined_stats.json"), "w") as f:
        json.dump(combined_stats.to_dict(), f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dsname",required=True,help="name of the generated dataset")
    ARGS = parser.parse_args()

    main(ARGS.dsname)

