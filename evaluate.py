import argparse
import contextlib
import datetime
import glob
import json
import logging
import os
import time
from pathlib import PurePath
from pprint import pprint

import optuna
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import ray
import tensorflow as tf
from scipy.optimize import linear_sum_assignment
from skimage.feature import peak_local_max
from sklearn.metrics import pairwise_distances
from tensorflow import keras
from tensorflow.keras import backend as K
from tqdm import tqdm
from sklearn.cluster import DBSCAN, KMeans, MiniBatchKMeans

from src import ARGS, DATA_DIR, MODEL_SAVE_FMT, REPO_ROOT
from src.eval_utils import (OVERLAP_METHODS, find_local_maxima, pointmatch,
                            rasterize_preds, viz_predictions)
from src.losses import get_loss
from src.models import pointnet
from src.patch_generator import get_datasets
from src.tf_utils import load_saved_model
from src.utils import (MyTimer, glob_modeldir, group_by_composite_key,
                       load_params_into_ARGS, plot_one_example,
                       rasterize_and_plot, rasterize_pts_gaussian_blur,
                       scaled_0_1, LabeledBox)



def clustering_postprocessing(pred_dict, algorithm, cluster_aggs):
    """
    args:
        pred_dict
        algo: no-args function that returns an sklearn clustering object
        cluster_aggs: list of str, options are "max" and/or "mean"
    returns:
        list of 2-tuples, where each tuple is:
            (pts, dict of best gridsearch params)
    """
    centroids_dict = {}
    exemplars_dict = {}
    for p_id, xyz in pred_dict.items():
        xy = xyz[:,:2]
        confs = xyz[:,-1]

        # initialize clean clustering and predict
        labels = algorithm().fit_predict(xy, sample_weight=confs)

        xyz = xyz[labels >= 0]
        labels = labels[labels >= 0]

        # two methods of finding the representative point in a cluster
        centroids = []
        exemplars = []
        for label in np.unique(labels):
            selected = xyz[labels == label]
            centroids.append(
                np.average(selected, axis=0, weights=selected[:,2])
            )
            exemplars.append(
                selected[np.argmax(selected[:,2])]
            )

        centroids_dict[p_id] = np.array(centroids)
        exemplars_dict[p_id] = np.array(exemplars)
    
    results = []
    if "mean" in cluster_aggs:
        results.append( (centroids_dict, {"cluster_agg": "mean"}) )
    if "max" in cluster_aggs:
        results.append( (exemplars_dict, {"cluster_agg": "max"}) )   
    assert len(results) 
    return results



def peaklocalmax_postprocessing(pred_dict, bounds_dict, grid_aggs, min_dists, min_conf_threshold):
    """
    args:
        pred_dict
        bounds_dict
        grid_aggs: rasterization aggregation modes
        min_dists: list of values to try for min_dists between peaks
        min_conf_threshold: lowest confidence threshold
    """
    pred_grids, pred_coords = rasterize_preds(pred_dict, bounds_dict, grid_aggs=grid_aggs, is_subdiv=False)

    results = []
    for grid_agg, these_grids in pred_grids.items():
        for min_dist in min_dists:
            pred_peaks = find_local_maxima(these_grids, pred_coords, min_dist=min_dist, conf_threshold=min_conf_threshold)
            results.append( (pred_peaks, {"min_dist": min_dist, "grid_agg": grid_agg}) )

    return results


def filter_by_conf_threshold(preds_dict, threshold):
    preds_dict = {
        p_id: pts[pts[:,2] > threshold] if len(pts) else pts
        for p_id, pts in preds_dict.items()
    }
    return {p_id: pts for p_id, pts in preds_dict.items() if pts.size}

def postprocess_and_pointmatch(preds, gt, bounds, params, gridsearch_params):
    """
    args:
        preds: dict mapping patch id to raw pred pts
        gt: dict mapping patch id to gt trees
        bounds: dict mapping patch id to bounds
        params: postprocessing method params
        gridsearch_params: secondary params which are gridsearched over (max vs summing, conf thresholds). 
            retuired keys: post_threshold
    returns:
        best pointmatch metrics
        best gridsearch params
    """
    timer = MyTimer()

    # Post-processing of raw predicted points to get final predicted tree locations
    postprocess_mode = params["postprocess_mode"]

    if postprocess_mode != "raw":
        orig_len = sum(map(len, preds.values()))
        pre_threshold = 10 ** params["pre_threshold_exp"]
        preds = filter_by_conf_threshold(preds, pre_threshold)
        new_len = sum(map(len, preds.values()))
        print("  pre-thresholding filtered {}% of points".format(round((orig_len - new_len) / orig_len * 100), 2))

    if postprocess_mode == "raw":
        processed_preds = [(preds, {})]
    elif postprocess_mode == "peaklocalmax":
        processed_preds = peaklocalmax_postprocessing(preds, bounds,
                                grid_aggs=gridsearch_params["grid_agg"],
                                min_dists=gridsearch_params["min_dist"],
                                min_conf_threshold=min(gridsearch_params["post_threshold"]))

    elif postprocess_mode == "dbscan":
        algo_initializer = lambda: DBSCAN(eps=params["eps"], min_samples=params["min_samples"])
        processed_preds = clustering_postprocessing(preds, algo_initializer,
                                cluster_aggs=gridsearch_params["cluster_agg"])

    elif postprocess_mode == "kmeans":
        if params["minibatch"]:
            algo_f = MiniBatchKMeans
        else:
            algo_f = KMeans
        algo_initializer = lambda: algo_f(n_clusters=params["n_clusters"], random_state=0)
        processed_preds = clustering_postprocessing(preds, algo_initializer, 
                                cluster_aggs=gridsearch_params["cluster_agg"])

    else:
        raise ValueError(f"Unknown postprocess mode {postprocess_mode}")

    timer.measure(f"postprocessing={postprocess_mode}")

    # Pointmatching on the results
    best_pointmatch_results = {"fscore": -1}
    best_gridparams = None
    for (pts_dict, these_gridparams) in processed_preds:
        # gridsearch through thresholds on post-processed pts
        for post_threshold in gridsearch_params["post_threshold"]:
            thresholded_pts = filter_by_conf_threshold(pts_dict, post_threshold)
            pointmatch_results = pointmatch(gt, thresholded_pts)
            if pointmatch_results["fscore"] > best_pointmatch_results["fscore"]:
                best_pointmatch_results = pointmatch_results
                best_gridparams = {
                    "post_threshold": post_threshold,
                    **these_gridparams
                }

    timer.measure(f"pointmatching x{len(processed_preds) * len(gridsearch_params['post_threshold'])}")
    
    return best_pointmatch_results, best_gridparams
    

# log distribution
ALL_POST_THRESHOLDS = list(10 ** np.arange(-5, 0.2, step=0.2))
# in units of 0.6-meter NAIP pixels
ALL_MIN_DISTS = [1, 2, 3, 4, 5]

def build_postprocessing_objective(raw_preds, gt, bounds, min_dists, post_thresholds, cache={}):
    """
    args:
        raw_preds: dict
        gt: dict
        min_dists: list of min dists to gridsearch
        post_thresholds: list of thresholds applying to post-processed preds to test
    """

    def objective(trial):
        # params for which evaluating a change in them is expensive
        params = {}
        # params for which we can evaluate all of them every time because it is quick to do
        gridparams = {
            "post_threshold": post_thresholds # threshold on post-processed peaks
        }

        # Post-processing of raw predicted points to get final predicted tree locations
        postprocess_mode = trial.suggest_categorical("postprocess_mode", ["peaklocalmax", "dbscan", "kmeans", "raw"])
        params["postprocess_mode"] = postprocess_mode
        if postprocess_mode == "raw":
            pass
        elif postprocess_mode == "peaklocalmax":
            params["gaussian_sigma"] = trial.suggest_float("gaussian_sigma", ARGS.gaussian_sigma-1.0, ARGS.gaussian_sigma+1.0, step=0.5) # in meters
            gridparams["min_dist"] = min_dists
            gridparams["grid_agg"] = ["max", "sum"]
        elif postprocess_mode == "dbscan":
            params["eps"] = trial.suggest_float("eps", 0.25, 5.0, step=0.25) # max distance between neighbors, in meters
            params["min_samples"] = trial.suggest_float("min_samples", 0, 10, step=0.2) # min total confidence in cluster
        elif postprocess_mode == "kmeans":
            #params["minibatch"] = trial.suggest_categorical("minibatch", [False, True])
            params["minibatch"] = True # regular kmeans is too slow
            params["n_clusters"] = trial.suggest_int("n_cluster", 10, 120, step=10)
        else:
            raise ValueError(f"Unknown postprocess mode {postprocess_mode}")

        if postprocess_mode in ("dbscan", "kmeans"):
            gridparams["cluster_agg"] = ["sum", "max"]

        if postprocess_mode != "raw":
            # threshold on raw points confs
            params["pre_threshold_exp"] = trial.suggest_float("pre_threshold_exp", -5, 0, step=0.2)
            # with 'raw' postprocessing, we just use the post_threshold instead of both, since they are redundant in that case

        print("Trial", trial.number, params)

        results, best_gridparams = postprocess_and_pointmatch(raw_preds, gt, bounds, params, gridparams)

        for key,value in best_gridparams.items():
            trial.set_user_attr(key, value)

        print(results)
        print(best_gridparams)

        return results["fscore"]

    return objective




def evaluate_pointmatching(patchgen, model, model_dir, pointmatch_thresholds):
    """
    generate predictions, visualizations, and evaluation metrics from a model 
    on a LidarPatchGen dataset
    args:
        patchgen: dataset, patch generator
        model: Keras Model to predict with
        model_dir: pathlib.PurePath of model's output
        pointmatch_thresholds: list of floats
    """
    outdir = model_dir.joinpath("results_"+patchgen.name)
    os.makedirs(outdir, exist_ok=True)

    timer = MyTimer()
    """
    generate predictions
    """
    print("\nGenerating predictions...")
    assert patchgen.batch_size == 1
    
    X_subdiv_normed, _ = patchgen.load_all()
    X_subdiv_normed = np.squeeze(X_subdiv_normed.numpy())
    preds_subdiv_normed = np.squeeze(model.predict(X_subdiv_normed))

    # associate each pred set with its patch id
    preds_subdiv_normed = dict(zip(patchgen.valid_patch_ids, preds_subdiv_normed))
    X_subdiv_normed = dict(zip(patchgen.valid_patch_ids, X_subdiv_normed))
    timer.measure()

    # denormalize data
    print("Denormalizing preds...")
    preds_subdiv_unnormed = {p_id: patchgen.denormalize_pts(pts, p_id) for p_id,pts in preds_subdiv_normed.items()}
    timer.measure()

    # combine with overlap
    print("Overlapping preds...")
    overlap_fn = OVERLAP_METHODS[ARGS.overlap_mode]
    preds_full_unnormed = overlap_fn(preds_subdiv_unnormed, patchgen.bounds_subdiv)
    timer.measure()
    
    # save predictions
    if ARGS.save_preds:
        preds_string_keys = {
            "_".join(map(str, p_id)): pts for p_id, pts in preds_full_unnormed.items()
        }
        outfile = outdir.joinpath("raw_preds.npz")
        np.savez_compressed(outfile.as_posix(), **preds_string_keys)
        del preds_string_keys


    sampler = optuna.samplers.TPESampler(
                multivariate=True,
                group=True,
                constant_liar=True,
                warn_independent_sampling=True,
                n_startup_trials=6, # 4 will be pre-set, 2 will be random
                seed=0)
    study = optuna.create_study(
                sampler=sampler,
                study_name="eval",
                direction="maximize", 
            )
    # ensure it tries all three methods at reasonable params
    study.enqueue_trial({"postprocess_mode": "raw"})
    study.enqueue_trial({"postprocess_mode": "dbscan", "eps": 1.0, "min_samples": 1.0, "pre_threshold": 1e-3})
    study.enqueue_trial({"postprocess_mode": "kmeans", "n_cluster": 100, "pre_threshold": 1e-3})
    study.enqueue_trial({"postprocess_mode": "peaklocalmax", "gaussian_sigma": ARGS.gaussian_sigma, "pre_threshold": 1e-3})
    objective = build_postprocessing_objective(preds_full_unnormed, patchgen.gt_full, 
                    patchgen.bounds_full, min_dists=ALL_MIN_DISTS, 
                    post_thresholds=ALL_POST_THRESHOLDS)
    study.optimize(objective, n_trials=200, timeout=60*10)

    best = study.best_trial
    params = best.params
    gridparams = best.user_attrs

    if not ARGS.noplot:
        print("Denormalizing & overlapping X...")
        # denormalize X
        X_subdiv_unnormed = {p_id: patchgen.denormalize_pts(pts, p_id) for p_id,pts in X_subdiv_normed.items()}

        # drop overlapping subpatches, combine into full patches
        X_full_unnormed = drop_overlaps(X_subdiv_unnormed)
        timer.measure()

        Y_subdiv_normed = patchgen.gt_subdiv
        viz_predictions(patchgen, outdir.joinpath("visualizations"), 
            # use normed for subdiv data
            X_subdiv=X_subdiv_normed, Y_subdiv=Y_subdiv_normed, preds_subdiv=preds_subdiv_normed, 
            # use unnormed for full data
            X_full=X_full_unnormed, Y_full=Y_full_unnormed, preds_full=preds_full_unnormed, 
            pred_grids=pred_grids_unnormed, pred_peaks=pred_peaks_unnormed)
        timer.measure()


    """
    Pointmatching
    """
    print("Pointmatching...")
    results = []
    for thresh in tqdm(pointmatch_thresholds):
        result = pointmatch(Y_full_unnormed, pred_peaks_unnormed, thresh)
        results.append(result)
    timer.measure()

    # find best, by fscore
    best_idx = np.argmax([x["fscore"] for x in results])

    best_stats = {k:results[best_idx][k] for k in results[0].keys()}
    all_stats = {k:[x[k] for x in results] for k in results[0].keys()}

    print("results:")
    pprint(best_stats)

    pointmatch_stats = {
        "best": best_stats,
        "all": all_stats,
    }

    # get or initialize stats for all overlap modes
    results_file = outdir.joinpath("results_pointmatch.json").as_posix()
    if os.path.exists(results_file):
        with open(results_file, "r") as f:
            all_modes_results = json.load(f)
    else:
        all_modes_results = {}
    # update with new stats
    all_modes_results[ARGS.overlap_mode] = pointmatch_stats
    # save
    with open(results_file, "w") as f:
        json.dump(all_modes_results, f, indent=2)

    return pointmatch_stats


def evaluate_loss_metrics(patchgen, model, model_dir):
    """
    Evaluate model's builtin metrics on a dataset
    args:
        patchgen
    """
    print("Evaluating model's metrics...")
    outdir = model_dir.joinpath("results_"+patchgen.name)
    os.makedirs(outdir, exist_ok=True)

    metric_vals = model.evaluate(patchgen, batch_size=ARGS.batchsize)
    
    if not isinstance(metric_vals, list):
        results = {"loss": metric_vals}
    else:
        results = {model.metrics_names[i]:v for i,v in enumerate(metric_vals)}

    with open(outdir.joinpath("results_model_metrics.json"), "w") as f:
        json.dump(results, f, indent=2)

    print("Results on", patchgen.name, "dataset:")
    for k,v in results.items():
        print(k+":", v)
    
    return results


def main():
    MODEL_DIR = glob_modeldir(ARGS.name)
    load_params_into_ARGS(MODEL_DIR, ARGS, false_params=["test", "noplot"])
    pprint(vars(ARGS))

    """
    Evaluation
    """

    model = load_saved_model(MODEL_DIR)

    if "test" in ARGS.eval_sets and "val" not in ARGS.eval_sets:
        ARGS.eval_sets += ("val",)
    datasets = get_datasets(ARGS.dsname, ARGS.regions, ARGS.eval_sets, train_batchsize=1, val_batchsize=1)
    datasets = dict(zip(ARGS.eval_sets, datasets))

    # grid search approximating log-scale
    THRESHOLDS = list(np.arange(0, 0.030, 0.005)) + list(np.arange(0.030, 0.100, 0.010)) + list(np.arange(0.100, 1, 0.025))
    # RASTERIZE_MODES = ["sum", "max"]

    # train set evaluation
    if "train" in ARGS.eval_sets:
        train_gen = datasets["train"]
        train_gen.training = False
        train_gen.summary()
        if not ARGS.nolosses:
            evaluate_loss_metrics(train_gen, model, MODEL_DIR)
        evaluate_pointmatching(train_gen, model, MODEL_DIR, THRESHOLDS)

    # validation set evaluation
    if "val" in ARGS.eval_sets or "test" in ARGS.eval_sets:
        val_gen = datasets["val"]
        val_gen.summary()
        if not ARGS.nolosses:
            evaluate_loss_metrics(val_gen, model, MODEL_DIR)
        pointmatch_stats = evaluate_pointmatching(val_gen, model, MODEL_DIR, THRESHOLDS)
        val_best_thresh = pointmatch_stats["best"]["threshold"]

    # test set evaluation
    if "test" in ARGS.eval_sets:
        test_gen = datasets["test"]
        test_gen.summary()
        if not ARGS.nolosses:
            evaluate_loss_metrics(test_gen, model, MODEL_DIR)
        evaluate_pointmatching(test_gen, model, MODEL_DIR, [val_best_thresh])




if __name__ == "__main__":
    """
    Parse arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--name",required=True,help="name of model to run, with possible timestamp in front")
    parser.add_argument("--sets",dest="eval_sets",default=("val","test"),nargs="+",help="datasets to evaluate on: train, val, and/or test. test automatically selects val as well")
    parser.add_argument("--noplot",action="store_true",help="don't make plots (significantly improves speed)")
    parser.add_argument("--nolosses",action="store_true",help="do not compute loss metrics")
    parser.add_argument("--overlap-mode",default="drop",choices=list(OVERLAP_METHODS.keys()),
        help="method by which to combine overlapping tiles")
    parser.add_argument("--save-preds",action="store_true",help="save raw predictions as npy files")
    parser.parse_args(namespace=ARGS)

    main()
