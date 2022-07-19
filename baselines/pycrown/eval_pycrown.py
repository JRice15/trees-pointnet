import os
import json
import sys
import argparse
import time
import glob
from pathlib import PurePath

import geopandas as gpd
import numpy as np
import optuna
import pandas as pd
from tqdm import tqdm

from __init__ import ROOT
from run_pycrown import pycrown_predict_treetops

from common.pointmatch import pointmatch
from common.data_handling import get_data_splits, load_gt_trees

def make_objective(raster_dirs, ground_truth):

    def get_raster_path(region, patchnum, kind):
        return os.path.join(
            raster_dirs[region],
            f"{kind}_{patchnum}.tif"
        )

    def objective(trial):
        params = {
            "chm_smooth_pix": tf.suggest_int("chm_smooth_pix", 0, 15), # in pixels
            "plm_ws": tf.suggest_int("plm_ws", 0, 15),  # in pixels
            "plm_min_dist": tf.suggest_int("plm_min_dist", 0, 15),  # in pixels
            "plm_threshold_abs": tf.suggest_float("plm_threshold_abs", 0, 20, step=0.5), # height in meters
            "inbuf_m": tf.suggest_int("inbuf_m", 0, 15), # in pixels
            "cdl_th_tree": tf.suggest_float("cdl_th_tree", 0, 1),
            "cdl_th_seed": tf.suggest_float("cdl_th_seed", 0, 1),
            "cdl_th_crown": tf.suggest_float("cdl_th_crown", 0, 1),
            "cdl_th_maxcrown": tf.suggest_int("cdl_th_maxcrown", 1, 20), # width in meters
        }

        preds = {
            "orig": {},
            "corrected": {},
        }
        for p_id in tqdm(list(ground_truth.keys())):
            region, patchnum = p_id
            PC = pycrown_predict_treetops(
                    chm=get_raster_path(region, patchnum, "chm"),
                    dtm=get_raster_path(region, patchnum, "dtm"),
                    dsm=get_raster_path(region, patchnum, "dsm"),
                    outpath=None,
                    params=params)
            
            preds["orig"] = PC.trees["top"].to_numpy()
            preds["corrected"] = PC.trees["top_cor"].to_numpy()
        
        orig_fscore = pointmatch(ground_truth, preds["orig"])["fscore"]
        corr_fscore = pointmatch(ground_truth, preds["corrected"])["fscore"]

        print(orig_fscore, corr_fscore)
        return max(orig_fscore, corr_fscore)



def estimate_params(raster_dirs):
    """
    args:
        raster_dirs: dict mapping region to path of directory which holds rasters (chms, dsms, dtms)
    """
    train_ids, test_ids = get_data_splits(sets=("train+val", "test"))

    train_gt = load_gt_trees(train_ids)
    test_gt = load_gt_trees(test_ids)

    objective = make_objective(raster_dirs, train_gt)

    sampler = optuna.samplers.TPESampler(
        multivariate=True, # consider the relations between different parameters
        group=True,
        warn_independent_sampling=True,
        constant_liar=True, # avoid very similar param combinations being tried simultaneously
        n_startup_trials=30, # number of random sample trials to begin with
        seed=1234,
    )

    study = optuna.create_study(sampler=sampler, name="pycrown", direction="maximize")
    study.optimize(n_trials=1)

    print("\nBest trial:")
    print("fscore:", study.best_value)
    print(study.best_params)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--specs",required=True,help="json, mapping region names to directory paths which contains raster tifs")
    args = parser.parse_args()

    with open(args.specs, "r") as f:
        specs = json.load(f)

    estimate_params(specs)

