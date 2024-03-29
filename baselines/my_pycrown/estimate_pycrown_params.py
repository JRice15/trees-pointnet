import os
import json
import sys
import argparse
import time
import glob
from pprint import pprint
from pathlib import PurePath

import numpy as np
import optuna
from tqdm import tqdm
import ray

from run_pycrown import pycrown_predict_treetops
from pycrown.pycrown import NoTreesException

from __init__ import ROOT, MY_PYCROWN_DIR
from common.utils import MyTimer
from common.pointmatch import pointmatch
from common.data_handling import get_data_splits, load_gt_trees, load_naip, naip2ndvi, get_all_patch_ids
from common.visualization import make_marker_dict, plot_NAIP

def get_study_storage(ARGS):
    name = "study_pycrown"
    if ARGS.spectral:
        name += "_spectral"
    path = str(MY_PYCROWN_DIR.joinpath(name + ".db"))

    storage = optuna.storages.RDBStorage(
        url="sqlite:///" + path,
        engine_kwargs={"connect_args": {"timeout": 10}},
        heartbeat_interval=10, # heartbeat in seconds
    )

    return storage


def make_objective(raster_dirs, ground_truth, return_results=False, spectral=False):

    def get_raster_path(region, patchnum, kind):
        return os.path.join(
            raster_dirs[region],
            f"{region}_{kind}_{patchnum}.tif"
        )

    def predict_patch(patch_id, params, spectral_mask=None):
        region, patchnum = patch_id

        try:
            PC = pycrown_predict_treetops(
                    chm=get_raster_path(region, patchnum, "chm"),
                    dtm=get_raster_path(region, patchnum, "dtm"),
                    dsm=get_raster_path(region, patchnum, "dsm"),
                    outpath=None,
                    params=params,
                    spectral_mask=spectral_mask)
        except Exception as e:
            if not isinstance(e, NoTreesException):
                print(f"Exception predicting {region} {patchnum}:", e.__class__.__name__, str(e))
            return np.array([]), np.array([])
            
        top = np.array([(p.x, p.y) for p in PC.trees["top"]])
        top_cor = np.array([(p.x, p.y) for p in PC.trees["top_cor"]])
        return top, top_cor
    
    if not ray.is_initialized():
        ray.init()

    @ray.remote
    def ray_predict_patch(*args, **kwargs):
        return predict_patch(*args, **kwargs)

    timer = MyTimer()

    # get ndvi images
    if spectral:
        ALL_NDVI = {}
        for p_id in ground_truth.keys():
            ALL_NDVI[p_id] = naip2ndvi(load_naip(*p_id))

    def objective(trial):
        timer.start()
        params = {
            "chm_smooth_pix": trial.suggest_int("chm_smooth_pix", 1, 15), # in pixels
            "plm_ws": trial.suggest_int("plm_ws", 1, 15),  # in pixels
            "plm_min_dist": trial.suggest_int("plm_min_dist", 1, 15),  # in pixels
            "plm_threshold_abs": trial.suggest_float("plm_threshold_abs", 0, 20, step=0.5), # height in meters
            "inbuf_m": trial.suggest_int("inbuf_m", 0, 15), # in meters
            "cdl_th_tree": trial.suggest_float("cdl_th_tree", 0, 1, step=0.01),
            "cdl_th_seed": trial.suggest_float("cdl_th_seed", 0, 1, step=0.01),
            "cdl_th_crown": trial.suggest_float("cdl_th_crown", 0, 1, step=0.01),
            "cdl_th_maxcrown": trial.suggest_float("cdl_th_maxcrown", 10, 30, step=0.5), # width in meters
        }

        # build pred func
        if spectral:
            threshold = trial.suggest_float("ndvi_thresh", -1, 1, step=0.01)
            spectral_masks = {p_id: (ndvi >= threshold) for p_id,ndvi in ALL_NDVI.items()}
            def pred_func(p_id):
                return ray_predict_patch.remote(
                    p_id, params, spectral_mask=spectral_masks[p_id]
                )
        else:
            def pred_func(p_id):
                return ray_predict_patch.remote(
                    p_id, params, spectral_mask=None,
                )

        # print(params)

        patch_ids = list(ground_truth.keys())
        preds = ray.get( [pred_func(p_id) for p_id in patch_ids] )
        # unpack corrected/uncorrected
        top, top_cor = zip(*preds)
        # associate with respective patch ids
        top = dict(zip(patch_ids, top))
        top_cor = dict(zip(patch_ids, top_cor))

        orig_results = pointmatch(ground_truth, top, return_inds=return_results)
        corr_results = pointmatch(ground_truth, top_cor, return_inds=return_results)
        if return_results:
            orig_results, orig_inds = orig_results
            corr_results, corr_inds = corr_results
        orig_fscore = orig_results["fscore"]
        corr_fscore = corr_results["fscore"]

        if orig_fscore == corr_fscore:
            trial.set_user_attr("method", "both")
        elif orig_fscore > corr_fscore:
            trial.set_user_attr("method", "orig")
        else:
            trial.set_user_attr("method", "corr")
        fscore = max(orig_fscore, corr_fscore)

        timer.measure()
        if return_results:
            return fscore, orig_results, corr_results, top_cor, corr_inds
        return fscore

    return objective


def estimate_params(ARGS, raster_dirs):
    """
    args:
        raster_dirs: dict mapping region to path of directory which holds rasters (chms, dsms, dtms)
    """
    (train_ids,) = get_data_splits(sets=("train+val",))
    train_gt = load_gt_trees(train_ids)

    objective = make_objective(raster_dirs, train_gt, spectral=ARGS.spectral)

    storage = get_study_storage(ARGS)

    sampler = optuna.samplers.TPESampler(
        multivariate=True, # consider the relations between different parameters
        group=True,
        warn_independent_sampling=True,
        constant_liar=True, # avoid very similar param combinations being tried simultaneously
        n_startup_trials=30, # number of random sample trials to begin with
        seed=1234,
    )

    study = optuna.create_study(
                sampler=sampler, 
                storage=storage,
                study_name="pycrown", 
                direction="maximize",
                load_if_exists=True,
            )
    print(len(study.trials), "trials already completed")
    study.optimize(objective, n_trials=ARGS.ntrials)

    print("\nBest trial:")
    print("fscore:", study.best_value)
    print(study.best_params)



def evaluate_params(ARGS, raster_dirs):
    (test_ids,) = get_data_splits(sets=("test",))
    test_gt = load_gt_trees(test_ids)
    
    objective = make_objective(raster_dirs, test_gt, return_results=True, 
                    spectral=ARGS.spectral)

    storage = get_study_storage(ARGS)

    study = optuna.load_study(study_name="pycrown", storage=storage)

    print("Best trial:", study.best_trial.number, "of", len(study.trials))
    print("Best params:", study.best_params)
    print("Best train fscore (found so far):", study.best_value)

    method_name = "pycrown_spectral" if ARGS.spectral else "pycrown"
    dirname = "results_" + method_name
    outdir = MY_PYCROWN_DIR.joinpath(dirname)
    os.makedirs(outdir.joinpath("plots"), exist_ok=True)

    fscore, orig, corr, preds, pmatch_inds = objective(study.best_trial)
    print("Test-set fscore:", fscore)
    results = {"corrected": corr, "uncorrected": orig, "spectral": ARGS.spectral}
    pprint(results)
    with open(outdir.joinpath("results.json"), "w") as f:
        json.dump(results, f, indent=2)
    
    print("Plotting...")
    for p_id, pred in tqdm(list(preds.items())):
        markers = make_marker_dict(gt=test_gt[p_id], preds=pred, pointmatch_inds=pmatch_inds[p_id])
        patch_name = "_".join(map(str, p_id))
        filename = outdir.joinpath("plots/" + method_name + "_" + patch_name)
        plot_NAIP(filename, markers=markers, patch_id=p_id)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    grp = parser.add_mutually_exclusive_group(required=True)
    grp.add_argument("--train",action="store_true",help="run `ntrials` more trials of hyperoptimization training")
    grp.add_argument("--eval",action="store_true",help="evaluate the best-performing params on the test set")
    parser.add_argument("--specs",default="elcap_chm_specs.json",help="json, mapping region names to directory paths which contains raster tifs")
    parser.add_argument("--ntrials",type=int,default=100)
    parser.add_argument("--spectral",action="store_true")
    ARGS = parser.parse_args()

    with open(ARGS.specs, "r") as f:
        raster_dirs = json.load(f)

    if ARGS.train:
        estimate_params(ARGS, raster_dirs)
    elif ARGS.eval:
        evaluate_params(ARGS, raster_dirs)
    else:
        raise ValueError()
