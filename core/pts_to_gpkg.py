import os
import re
import sys
import json
import argparse
import glob
import traceback
from pprint import pprint
from pathlib import PurePath

import geopandas as gpd
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from core import REPO_ROOT, DATA_DIR
from core.utils import glob_modeldir, gaussian, gridify_pts, get_dataset_dir
from core import pointmatch

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
    dataset_dir, dsname = get_dataset_dir(patchgen.dsname)
    os.makedirs(dataset_dir.joinpath("gt_gpkgs"), exist_ok=True)
    for region in regions_data:
        with h5py.File(region["ds"], "r") as f:
            patch_ids = [i for i in patchgen.patch_ids if i[0] == region["name"]]
            for i,(region_name,patchname) in enumerate(patch_ids):
                outfile = dataset_dir.joinpath("gt_gpkgs/{}_{}_gt.gpkg".format(region_name, patchname))
                if not os.path.exists(outfile):
                    pts = f["/gt/"+patchname][:,:2]
                    x = pts[:,0]
                    y = pts[:,1]

                    df = gpd.GeoDataFrame(geometry=gpd.points_from_xy(x, y), crs=region["meta"]["crs"])

                    make_gpkg(df, outfile.as_posix())



def estimate_pred_thresh(patchgen, modeldir, step=0.05):
    print("Estimating optimal prediction threshold")
    thresholds = np.arange(0.2, 1, step)
    results = []
    for thresh in thresholds:
        r = evaluate_preds_to_gpkg(patchgen, modeldir, threshold=thresh)
        results.append(r["f1"])
        print(" ", thresh, "f1:", r["f1"])
    best = thresholds[np.argmax(results)]
    with open(modeldir.joinpath("results_{}/optimal_threshold.json".format(patchgen.name)), "w") as f:
        json.dump(best, f)
    return best


def evaluate_preds_to_gpkg(patchgen, modeldir, resolution=50, threshold=0.6):
    """
    from prediction locations, guassian blur them on a grid, and find localmax points
    that are above threshold
    args:
        sigma: gaussian sigma for rasterizations
        resolution: width in "pixels" of the blur grid
        threshold: min confidence to keep point
    """
    predfile = modeldir.joinpath("results_"+patchgen.name+"/predictions.npz")
    outdir = modeldir.joinpath("results_"+patchgen.name+"/pred_gpkgs/")
    os.makedirs(outdir, exist_ok=True)

    regions_data = get_regions_data(patchgen.dsname)
    dataset_dir, dsname = get_dataset_dir(patchgen.dsname)

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

    all_results = []
    for region in regions_data:
        norm_data = patchgen.norm_data[region["name"]]
        for i in range(len(patch_ids)):
            patch_region, patchname = patch_ids[i]
            patch_num = int(patchname[5:])
            if patch_region != region["name"]:
                continue
            pred_i = pred[i]

            realgrid, gridcoords = gridify_pts((0,1,0,1), pred_i[:,:2], pred_i[:,2], 
                                        gaussian_sigma=sigma, mode="sum", 
                                        resolution=resolution)
            grid = np.pad(realgrid, 1) # one row/col of zeros padding on all sides
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
        
            df = gpd.GeoDataFrame(geometry=gpd.points_from_xy(x, y), crs=region["meta"]["crs"])

            try:
                outfile = outdir.joinpath(patch_region+"_"+patchname+"_pred.gpkg").as_posix()
                make_gpkg(df, outfile)
                gtfile = dataset_dir.joinpath("gt_gpkgs/{}_{}_gt.gpkg".format(patch_region, patchname)).as_posix()
                results = pointmatch.main(gtfile, outfile)
            except Exception as e:
                print("Error:")
                traceback.print_exc()
                results = {
                    "recall": 0,
                    "f1": 0,
                    "precision": 1,
                    "rmse": np.nan,
                }
            print(patch_region, patchname, results)
            all_results.append(results)
    
    accum_results = {}
    for k in all_results[0].keys():
        accum_results[k] = 0
        for r in all_results:
            accum_results[k] += r[k]
        accum_results[k] /= len(all_results)
    
    return accum_results

