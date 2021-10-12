import os
import re
import sys
import json
import argparse
import glob
from pprint import pprint
from pathlib import PurePath

import geopandas as gpd
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from core import REPO_ROOT, DATA_DIR
from core.utils import glob_modeldir, gaussian, gridify_pts, get_dataset_dir

def get_regions_data(dsname):
    dataset_dir, dsname = get_dataset_dir(dsname)

    region_files = glob.glob(dataset_dir.joinpath("*.h5").as_posix())
    region_files = {PurePath(i).stem:PurePath(i) for i in region_files}
    region_metas = {}
    for region_name in region_files.keys():
        fname = PurePath(dataset_dir.joinpath(region_name + "_meta.json"))
        with open(fname, "r") as f:
            region_metas[region_name] = json.load(f)

    regions_data = [
        {"ds": region_files[region],
         "meta": region_metas[region],
         "name": region,
         "gt_outfile": region_files[region].parent.joinpath(region+"_gt.gpkg"),
        } for region in region_files.keys()
    ]
    return regions_data

def make_gpkg(df, filename):
    """
    args:
        df: geopandas dataframe
    """
    df = df.to_crs("EPSG:4326")

    df["Longitude"] = df.geometry.x
    df["Latitude"] = df.geometry.y

    df.to_file(filename, driver="GPKG")


def make_gt_gpkg(patchgen):
    regions_data = get_regions_data(patchgen.dsname)
    for region in regions_data:
        if not os.path.exists(region["gt_outfile"]):
            Xs = np.array([])
            Ys = np.array([])
            with h5py.File(region["ds"], "r") as f:
                patch_ids = [i for i in patchgen.patch_ids if i[0] == region["name"]]
                for i,(region_name,patchname) in enumerate(patch_ids):
                    pts = f["/gt/"+patchname][:,:2]
                    x = pts[:,0]
                    y = pts[:,1]
                    Xs = np.concatenate([Xs, x])
                    Ys = np.concatenate([Ys, y])

            df = gpd.GeoDataFrame(geometry=gpd.points_from_xy(Xs, Ys), crs=region["meta"]["crs"])

            make_gpkg(df, region["gt_outfile"].as_posix())


def make_pred_gpkg(patchgen, modeldir, predfile, outdir, resolution=50, threshold=0.05):
    """
    from prediction locations, guassian blur them on a grid, and find localmax points
    that are above threshold
    args:
        sigma: gaussian sigma for rasterizations
        resolution: width in "pixels" of the blur grid
        threshold: min confidence to keep point
    """
    os.makedirs(outdir, exist_ok=True)

    regions_data = get_regions_data(patchgen.dsname)

    with open(modeldir.joinpath("params.json"), "r") as f:
        params = json.load(f)
        if "mmd" not in params["mode"]:
            raise NotImplementedError()
        sigma = params["mmd_sigma"]
        pred_regions = params["regions"]

    regions_data = [i for i in regions_data if i["name"] in pred_regions]

    npz = np.load(predfile)
    pred = npz["pred"]
    patch_ids = npz["patch_ids"]

    for region in regions_data:
        norm_data = patchgen.norm_data[region["name"]]
        x_coords = []
        y_coords = []
        for i in range(len(patch_ids)):
            patch_region, patchname = patch_ids[i]
            patch_num = int(patchname[5:])
            if patch_region != region["name"]:
                continue
            pred_i = pred[i]

            realgrid, gridcoords = gridify_pts((0,1,0,1), pred_i[:,:2], pred_i[:,2], 
                                        gaussian_sigma=sigma, mode="sum", 
                                        resolution=resolution)
            grid = np.pad(realgrid, 1) # one row of zeros everywhere
            # mask of whether a pt is larger than all its neighbors
            localmax = (realgrid >= grid[2:,1:-1]) & (realgrid > grid[:-2,1:-1]) \
                        & (realgrid >= grid[1:-1,2:]) & (realgrid > grid[1:-1,:-2])
            # set non localmax elems to 0
            maxgrid = realgrid * localmax.astype(int)
            x_inds, y_inds = np.nonzero(maxgrid >= threshold)

            # get fractions from 0-1, convert to proper CRS coordinates
            pred_coords = gridcoords[x_inds, y_inds]
            min_xy = norm_data["min_xyz"][patch_num,:2]
            max_xy = norm_data["max_xyz"][patch_num,:2]
            unnormed_coords = (pred_coords * (max_xy - min_xy)) + (min_xy)
            x = unnormed_coords[...,0]
            y = unnormed_coords[...,1]
            x_coords = np.concatenate([x_coords, x])
            y_coords = np.concatenate([y_coords, y])
        
        df = gpd.GeoDataFrame(geometry=gpd.points_from_xy(x_coords, y_coords), crs=region["meta"]["crs"])

        make_gpkg(df, outdir.joinpath(region["name"]+"_pred.gpkg").as_posix())


