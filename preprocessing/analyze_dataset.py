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
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.cm as cm
import numpy as np
import pandas as pd
import seaborn

# add parent directory
dn = os.path.dirname
sys.path.append(dn(dn(os.path.abspath(__file__))))

from src import DATA_DIR, LIDAR_CHANNELS


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


def create_histplots(pts_df, region_name, outdir):
    for i,name in enumerate(LIDAR_CHANNELS):
        if name in ("x", "y"):
            continue
        # generate binned stats plots
        seaborn.histplot(data=pts_df, x=name, bins=12)
        plt.yscale("log")
        plt.ylabel("count"), plt.xlabel(name)
        # plt.ylim(bottom=1)
        plt.title(region_name + " " + name)
        plt.savefig(outdir.joinpath(name).as_posix())
        plt.clf()


NDVI_CHAN = LIDAR_CHANNELS.index("ndvi")


def main(dsname):
    print("Running analyze_dataset.py")
    all_stats = {}
    cumulative_df = None
    cumulative_pts = []

    DS_dir = DATA_DIR.joinpath("lidar", dsname, "regions")
    region_dirs = glob.glob(DS_dir.as_posix() + "/*/")

    for region_dir in region_dirs:
        region_name = PurePath(region_dir).stem
        print("Analyzing", region_name)

        outdir = DATA_DIR.joinpath("lidar", dsname, "stats", region_name)
        os.makedirs(outdir, exist_ok=True)

        patch_files = glob.glob(region_dir + "*.npy")

        # patchwise stats for the region
        patch_stats = []
        # bin_stats = []
        region_pts = []
        for file in patch_files:
            pts = np.load(file)
            region_pts.append(pts)
        
            patch_stats.append({
                "n_patches": 1,
                "n_points": pts.shape[0],
            })

        
        region_pts = np.concatenate(region_pts, axis=0)
        region_pts[region_pts < -1e30] = np.nan # no data values
        pts_df = pd.DataFrame(region_pts, columns=LIDAR_CHANNELS)
        create_histplots(pts_df, region_name, outdir=outdir)

        patch_df = pd.DataFrame(patch_stats)

        # generate patchwise plots
        seaborn.histplot(data=patch_df, x="n_points", bins=20, 
                binrange=(0,patch_df["n_points"].max()))
        plt.title(region_name + " points per patch")
        plt.savefig(outdir.joinpath("pount_counts").as_posix())
        plt.clf()

        # save region-cumulative stats
        region_stats = patch_df.sum(axis=0).to_dict()
        region_stats["no_spectral_data_pts"] = int(np.isnan(region_pts).any(axis=-1).sum())
        with open(outdir.joinpath("computed_stats.json"), "w") as f:
            json.dump(region_stats, f, indent=2)

        if cumulative_df is None:
            cumulative_df = patch_df
        else:
            cumulative_df = pd.concat([cumulative_df, patch_df])
        cumulative_pts.append(region_pts)

    print("Analyzing cumulative stats")

    outdir = DATA_DIR.joinpath("lidar", dsname, "stats")

    all_pts = np.concatenate(cumulative_pts, axis=0)
    all_pts_df = pd.DataFrame(all_pts, columns=LIDAR_CHANNELS)
    create_histplots(all_pts_df, "cumulative", outdir=outdir)

    # generate all-region patchwise plot
    seaborn.histplot(data=cumulative_df, x="n_points", bins=20, 
            binrange=(0,cumulative_df["n_points"].max()))
    plt.title("all-regions points per patch")
    plt.savefig(outdir.joinpath("pount_counts").as_posix())
    plt.clf()

    cumulative_stats = cumulative_df.sum(axis=0).to_dict()
    with open(outdir.joinpath("cumulative_stats.json"), "w") as f:
        json.dump(cumulative_stats, f, indent=2)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dsname",required=True,help="name of the lidar dataset")
    ARGS = parser.parse_args()

    main(ARGS.dsname)

