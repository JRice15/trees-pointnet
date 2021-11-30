import os
import re
import sys
import json
import argparse
import glob
import traceback
import shutil
from pprint import pprint
from pathlib import PurePath

import geopandas as gpd
import h5py
import ray
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
    ray.init()
    thresholds = np.arange(0.15, 1, step)
    results = []
    for thresh in thresholds:
        r = test_thresh.remote(patchgen.name, patchgen.dsname, patchgen.norm_data, modeldir, threshold=thresh)
        results.append(r)
    results = ray.get(results)
    f1s = [i["f1"] for i in results]
    best_f1 = np.max(f1s)
    best_thresh = thresholds[np.argmax(f1s)]
    print("  (threshold, f1) pairs:", list(zip(thresholds, f1s)))
    print("  best threshold:", best_thresh, "f1:", best_f1)
    dct = {
        "best_thresh": best_thresh,
        "thresholds": thresholds.tolist(),
        "f1s": f1s
    }
    with open(modeldir.joinpath("results_{}/optimal_threshold.json".format(patchgen.name)), "w") as f:
        json.dump(dct, f, indent=2)
    outdir = modeldir.joinpath("results_"+patchgen.name+"/temp_pred_gpkgs")
    shutil.rmtree(outdir.as_posix())
    return best_thresh

@ray.remote
def test_thresh(pg_name, pg_ds_name, pg_norm_data, *args, **kwargs):
    """
    Spin up one worker to test a threshold value
    """
    print("  testing threshold:", kwargs["threshold"])
    # patch generator is not pickleable, so it can't be passed here. But we only need these few attributes from it
    class dummy_patchgen:
        name = pg_name
        dsname = pg_ds_name
        norm_data = pg_norm_data
    out_subdir = "temp_pred_gpkgs/test_"+str(kwargs["threshold"])+"/"
    output = evaluate_preds_to_gpkg(dummy_patchgen, *args, **kwargs, out_subdir=out_subdir)
    return output

def evaluate_preds_to_gpkg(patchgen, modeldir, resolution=50, threshold=0.6, 
        save=False, out_subdir="pred_gpkgs"):
    """
    from prediction locations, guassian blur them on a grid, and find localmax points
    that are above threshold
    args:
        sigma: gaussian sigma for rasterizations
        resolution: width in "pixels" of the blur grid
        threshold: min confidence to keep point
    """
    predfile = modeldir.joinpath("results_"+patchgen.name+"/predictions.npz")
    outdir = modeldir.joinpath("results_"+patchgen.name+"/"+out_subdir)
    os.makedirs(outdir, exist_ok=True)

    regions_data = get_regions_data(patchgen.dsname)
    dataset_dir, dsname = get_dataset_dir(patchgen.dsname)

    with open(modeldir.joinpath("params.json"), "r") as f:
        params = json.load(f)
        if params["output_mode"] not in ("seg", "dense"):
            raise NotImplementedError()
        sigma = params["mmd_sigma"]
        pred_regions = params["regions"]

    regions_data = [i for i in regions_data if i["name"] in pred_regions]

    npz = np.load(predfile)
    pred = npz["pred"]
    patch_ids = npz["patch_ids"]

    all_results = []
    successes = 0
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

            outfile = outdir.joinpath(patch_region+"_"+patchname+"_pred.gpkg").as_posix()
            make_gpkg(df, outfile)
            gtfile = dataset_dir.joinpath("gt_gpkgs/{}_{}_gt.gpkg".format(patch_region, patchname)).as_posix()

            try:
                results = pointmatch.main(gtfile, outfile)
                successes += 1
            except AssertionError as e:
                results = {
                    "recall": 0,
                    "f1": 0,
                    "precision": 1,
                    "rmse": np.nan,
                }
            all_results.append(results)
    
    accum_results = {}
    for k in all_results[0].keys():
        accum_results[k] = 0
        for r in all_results:
            accum_results[k] += r[k]
        accum_results[k] /= len(all_results)
    
    if save:
        resultsfile = modeldir.joinpath("results_"+patchgen.name+"/pointmatch_results.json")
        output = {
            "results": accum_results,
            "pointmatch_success_rate": successes / len(all_results),
            "threshold": threshold
        }
        with open(resultsfile.as_posix(), "w") as f:
            json.dump(output, f, indent=2)

    return accum_results

