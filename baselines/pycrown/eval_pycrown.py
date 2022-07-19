import os
import sys
import time

import geopandas as gpd
import numpy as np
import optuna
import pandas as pd
from tqdm import tqdm

from __init__ import ROOT
from run_pycrown import pycrown_predict_treetops

from common.pointmatch import pointmatch


def make_objective(chms, dtms, dsms):
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

        outputs = {
            "orig": {},
            "corrected": {},
        }
        for key in tqdm(chms.keys()):
            PC = pycrown_predict_treetops(
                    chm=chms[key],
                    dtm=dtms[key],
                    dsm=dsms[key],
                    outpath=None,
                    params=params)
            
            outputs["orig"] = PC.trees["top"].to_numpy()
            outputs["corrected"] = PC.trees["top_cor"].to_numpy()
        
        pointmatch(all_gts, all_preds)



def estimate_params():
    sampler = optuna.samplers.TPESampler(
        multivariate=True, # consider the relations between different parameters
        group=True,
        warn_independent_sampling=True,
        constant_liar=True, # avoid very similar param combinations being tried simultaneously
        n_startup_trials=30, # number of random sample trials to begin with
        seed=1234,
    )

    objective = make_objective(chms, dtms, dsms)

    study = optuna.create_study(sampler=sampler, name="pycrown", direction="maximize")
    study.optimize(n_trials=200)



if __name__ == "__main__":
    estimate_params()

