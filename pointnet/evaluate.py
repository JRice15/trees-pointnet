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

from src import ARGS, DATA_DIR
from src.eval_utils import (OVERLAP_METHODS, find_local_maxima,
                            rasterize_preds, viz_predictions)
from src.losses import get_loss
from src.models import pointnet
from src.patch_generator import get_datasets
from src.tf_utils import load_saved_model
from src.utils import glob_modeldir, load_params_into_ARGS
from src.viz_utils import rasterize_pts_gaussian_blur, plot_one_example

from common.pointmatch import pointmatch
from common.utils import MyTimer

def clustering_postprocessing(pred_dict, algo_initializer, cluster_aggs):
    """
    args:
        pred_dict
        algo_initializer: no-args function that returns an sklearn clustering object
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
        algorithm = algo_initializer()
        try:
            labels = algorithm.fit_predict(xy, sample_weight=confs)
        except ValueError as e:
            # sometimes things go wrong, like trying to cluster less than n_clusters points with kmeans
            print("Clustering error:", e)
            raise optuna.exceptions.TrialPruned()

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


def postprocess_and_pointmatch(preds_overlapped, gt, bounds, params, gridsearch_params, 
        return_inds=False):
    """
    args:
        preds_overlapped: dict mapping overlap method to dict mapping patch id to raw pred pts
        gt: dict mapping patch id to gt trees
        bounds: dict mapping patch id to bounds
        params: postprocessing method params
        gridsearch_params: secondary params which are gridsearched over (max vs summing, conf thresholds). 
            required keys: post_threshold
        return_inds: return tp/fp/fn indexes in output dict
    returns:
        dict with keys:
            metrics: best pointmatch metrics
            gridparams: best gridsearch params
            pts: final pred_dict
            (if return_inds) inds: tp/fp/fn indexes
    """
    timer = MyTimer()

    # get correctly overlapped preds
    preds = preds_overlapped[params["overlap_method"]]

    # Post-processing of raw predicted points to get final predicted tree locations
    postprocess_mode = params["postprocess_mode"]

    # pre-thresholding for most methods
    if postprocess_mode != "raw":
        orig_len = sum(map(len, preds.values()))
        pre_threshold = 10 ** params["pre_threshold_exp"]
        preds = filter_by_conf_threshold(preds, pre_threshold)
        new_len = sum(map(len, preds.values()))

    # Post processing
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

    # Pointmatching on the results
    best_pointmatch_results = {"fscore": -1}
    best_gridparams = None
    best_pts = None
    best_inds = None
    # post-processing methods can return multiple options internally, if they do a gridsearch inside the method somewhere
    for (pts_dict, these_gridparams) in processed_preds:
        # gridsearch through thresholds on post-processed pts
        for post_threshold in gridsearch_params["post_threshold"]:
            # post-thresholding
            thresholded_pts = filter_by_conf_threshold(pts_dict, post_threshold)
            # run pointmatching
            pointmatch_results = pointmatch(gt, thresholded_pts, return_inds=return_inds)
            if return_inds:
                pointmatch_results, pointmatch_inds = pointmatch_results
            # save if we found new best fscore
            if pointmatch_results["fscore"] > best_pointmatch_results["fscore"]:
                best_pointmatch_results = pointmatch_results
                best_gridparams = {
                    "post_threshold": post_threshold,
                    **these_gridparams
                }
                best_pts = thresholded_pts
                if return_inds:
                    best_inds = pointmatch_inds
    
    timer.measure()

    output = {
        "metrics": best_pointmatch_results,
        "gridparams": best_gridparams,
        "pts": best_pts
    }
    if return_inds:
        output["inds"] = best_inds
    return output
    


def build_postprocessing_objective(preds_overlapped, gt, bounds, min_dists, post_thresholds):
    """
    args:
        preds_overlapped: dict mapping overlap method to dict mapping p_id to pts
        gt: dict mapping p_id to pts
        bounds: dict mapping p_id to Bounds
        min_dists: list of min dists to gridsearch
        post_thresholds: list of thresholds applying to post-processed preds to test
    """

    def objective(trial):
        # params for which evaluating a change in them is expensive
        params = {
            # Post-processing of raw predicted points to get final predicted tree locations
            "postprocess_mode": trial.suggest_categorical("postprocess_mode", ["peaklocalmax", "dbscan", "kmeans", "raw"]),
            # how to combine overlapping tiles
            "overlap_method": trial.suggest_categorical("overlap_method", list(OVERLAP_METHODS.keys()))
        }
        # params for which we can evaluate all of them every time because it is quick to do
        gridparams = {
            "post_threshold": post_thresholds # threshold on post-processed peaks
        }

        postprocess_mode = params["postprocess_mode"]
        if postprocess_mode == "raw":
            pass
        elif postprocess_mode == "peaklocalmax":
            params["gaussian_sigma"] = trial.suggest_float("gaussian_sigma", 0.5, 10, step=0.5) # in meters
            gridparams["min_dist"] = min_dists
            gridparams["grid_agg"] = ["max", "sum"]
        elif postprocess_mode == "dbscan":
            params["eps"] = trial.suggest_float("eps", 0.1, 5.0, step=0.1) # max distance between neighbors, in meters
            params["min_samples"] = trial.suggest_float("min_samples", 0, 0.5, step=0.01) # min total confidence in cluster
        elif postprocess_mode == "kmeans":
            #params["minibatch"] = trial.suggest_categorical("minibatch", [False, True])
            params["minibatch"] = True # regular kmeans is too slow
            params["n_clusters"] = trial.suggest_int("n_cluster", 5, 150, step=5)
        else:
            raise ValueError(f"Unknown postprocess mode {postprocess_mode}")

        if postprocess_mode in ("dbscan", "kmeans"):
            gridparams["cluster_agg"] = ["sum", "max"]

        if postprocess_mode != "raw":
            # threshold on raw points confs
            params["pre_threshold_exp"] = trial.suggest_float("pre_threshold_exp", -5, 0, step=0.2)
            # with 'raw' postprocessing, we just use the post_threshold instead of both, since they are redundant in that case

        results = postprocess_and_pointmatch(preds_overlapped, gt, bounds, params, gridparams, return_inds=False)

        for key,value in results["gridparams"].items():
            trial.set_user_attr(key, value)

        return results["metrics"]["fscore"]

    return objective



def generate_predictions(patchgen, model, outdir):
    """
    generate, denormalize, and overlap predictions
    returns:
        overlapped preds: dict mapping overlap_method to dict mapping p_id to pts
        X_subdiv_normed: raw input points
    """
    os.makedirs(outdir, exist_ok=True)

    print("\nGenerating predictions...")
    timer = MyTimer()

    X_subdiv_normed, _ = patchgen.load_all()
    X_subdiv_normed = np.squeeze(X_subdiv_normed.numpy())
    preds_subdiv_normed = np.squeeze(model.predict(X_subdiv_normed))

    # associate each pred set with its patch id
    preds_subdiv_normed = dict(zip(patchgen.valid_patch_ids, preds_subdiv_normed))
    X_subdiv_normed = dict(zip(patchgen.valid_patch_ids, X_subdiv_normed))
    timer.measure("generation")

    # denormalize data
    print("Denormalizing preds...")
    preds_subdiv_unnormed = {}
    for p_id, weighted_pts in preds_subdiv_normed.items():
        # only denormalize the locs, the confs are something else
        pts = weighted_pts[...,:2]
        confs = weighted_pts[...,2:]
        pts = patchgen.denormalize_pts(pts, p_id)
        preds_subdiv_unnormed[p_id] = np.concatenate((pts, confs), axis=-1)
    timer.measure("denormalize")

    # combine with overlap
    print("Overlapping preds...")
    overlapped_preds = {}
    for name, overlap_fn in OVERLAP_METHODS.items():
        preds_overlapped = overlap_fn(preds_subdiv_unnormed, patchgen.bounds_subdiv)
        overlapped_preds[name] = preds_overlapped
    timer.measure("overlap")

    return overlapped_preds, X_subdiv_normed



def estimate_postproc_params(preds_overlapped_dict, gt_dict, bounds_dict, outdir):
    """
    Estimate best postprocessing params on a set of predictions and their corresponding ground truth
    returns:
        params
        gridparams
    """
    # some values to gridsearch over
    # log distribution
    ALL_POST_THRESHOLDS = list(10 ** np.arange(-4, 0.1, step=0.1))
    # in units of 0.6-meter NAIP pixels
    ALL_MIN_DISTS = [1, 2, 3, 4, 5]

    sampler = optuna.samplers.TPESampler(
                multivariate=True,
                group=True,
                constant_liar=True,
                warn_independent_sampling=True,
                n_startup_trials=25, # 4 will be pre-set, 21 will be random
                seed=0)
    study = optuna.create_study(
                sampler=sampler,
                study_name="eval",
                direction="maximize",
            )
    # ensure it tries all methods at reasonable params
    study.enqueue_trial(
        {"postprocess_mode": "raw", "overlap_method": "buffer"})
    study.enqueue_trial(
        {"postprocess_mode": "raw", "overlap_method": "drop"})
    study.enqueue_trial(
        {"postprocess_mode": "dbscan", "eps": 1.0, "min_samples": 0.5, "pre_threshold_exp": -3, "overlap_method": "buffer"})
    study.enqueue_trial(
        {"postprocess_mode": "kmeans", "n_cluster": 100, "pre_threshold_exp": -3, "overlap_method": "buffer"})
    study.enqueue_trial(
        {"postprocess_mode": "peaklocalmax", "gaussian_sigma": ARGS.gaussian_sigma, "pre_threshold_exp": -3, "overlap_method": "buffer"})
    # force it to try all three methods equally after that (raw has already had all two parameter combinations tried)
    for i in range(50): # 3*50 future trials, more than is needed
        study.enqueue_trial({"postprocess_mode": "dbscan"})
        study.enqueue_trial({"postprocess_mode": "kmeans"})
        study.enqueue_trial({"postprocess_mode": "peaklocalmax"})

    objective = build_postprocessing_objective(
                    preds_overlapped_dict, gt_dict, bounds_dict, 
                    min_dists=ALL_MIN_DISTS, 
                    post_thresholds=ALL_POST_THRESHOLDS)


    # initial batch of trials
    n_trials = 5 if ARGS.test else 40
    study.optimize(objective, 
        n_trials=n_trials, # max trials, timeout supersedes
        timeout=(10*60), # 10 minute timeout
    )

    print(len(study.trials), "completed in initial batch batch")

    # second batch, if first are promising
    if study.best_value >= 0.50 and (not ARGS.test):
        print("promising, continuing...")
        study.optimize(objective,
            n_trials=100, # run 100 more trials, timeout supersedes
            timeout=(25*60), # 25 min timout
        )
        print(len(study.trials), "completed in total")

    study_dir = outdir.joinpath("postprocessing_param_estimation/")
    os.makedirs(study_dir, exist_ok=True)

    try:
        optuna.visualization.plot_optimization_history(study) \
            .write_image(study_dir.joinpath("optimization_history.png").as_posix(), scale=2)
    except Exception as e:
        print("could not make opt history plot:", str(e))
    try:
        optuna.visualization.plot_param_importances(study) \
            .write_image(study_dir.joinpath("param_importances.png").as_posix(), scale=2)
    except Exception as e:
        print("could not make param importance plot:", str(e))
    try:
        optuna.visualization.plot_slice(study) \
            .write_image(study_dir.joinpath("slice_plot.png").as_posix(), scale=2)
    except Exception as e:
        print("could not make slice plot:", str(e))

    best = study.best_trial
    params = best.params
    gridparams = best.user_attrs

    return params, gridparams


def evaluate_postproc_params(patchgen, outdir, preds_overlapped_dict, X_subdiv_normed, params, gridparams):
    """
    Evaluate the final postprocessing params found by optuna
    args:

    """
    timer = MyTimer()

    # gridparams expects a list for each value
    formatted_gridparams = {k:[v] for k,v in gridparams.items()}
    # run postprocessing
    results = postprocess_and_pointmatch(preds_overlapped_dict, 
                                patchgen.gt_full, patchgen.bounds_full, 
                                params=params, gridsearch_params=formatted_gridparams,
                                return_inds=True)
    timer.measure("final post-processing")

    # save stats
    results_file = outdir.joinpath("results_pointmatch.json").as_posix()
    all_stats = {
        "metrics": results["metrics"],
        "params": params,
        "gridparams": results["gridparams"],
    }
    with open(results_file, "w") as f:
        json.dump(all_stats, f, indent=2)

    # save predictions
    if ARGS.save_preds:
        preds_string_keys = {
            "_".join(map(str, p_id)): pts for p_id, pts in results["pts"].items()
        }
        outfile = outdir.joinpath("raw_preds.npz")
        np.savez_compressed(outfile.as_posix(), **preds_string_keys)
        del preds_string_keys

    if not ARGS.noplot:
        raw_preds = preds_overlapped_dict[params["overlap_method"]]
        print("Generating visualizations...")
        viz_predictions(
            patchgen, outdir.joinpath("visualizations"), 
            X_subdiv=X_subdiv_normed, 
            Y_subdiv=patchgen.gt_subdiv,
            Y_full=patchgen.gt_full, 
            bounds_full=patchgen.bounds_full,
            preds_full=raw_preds, 
            preds_full_peaks=results["pts"],
            pointmatch_inds=results["inds"],)
        timer.measure()

    return results



def evaluate_loss_metrics(patchgen, model, outdir):
    """
    Evaluate model's builtin metrics on a dataset
    args:
        patchgen
    """
    os.makedirs(outdir, exist_ok=True)

    print("Evaluating model's metrics...")
    metric_vals = model.evaluate(patchgen, batch_size=ARGS.batchsize)
    
    if not isinstance(metric_vals, list):
        results = {"loss": metric_vals}
    else:
        results = dict(zip(model.metrics_names, metric_vals))

    with open(outdir.joinpath("results_model_metrics.json"), "w") as f:
        json.dump(results, f, indent=2)

    print("Results on", patchgen.name, "dataset:")
    for k,v in results.items():
        print(k+":", v)
    
    return results



def main():
    timer = MyTimer()

    MODEL_DIR = glob_modeldir(ARGS.name)
    load_params_into_ARGS(MODEL_DIR, ARGS, false_params=["test", "noplot"])
    pprint(vars(ARGS))

    """
    Evaluation
    """

    model = load_saved_model(MODEL_DIR)

    val_gen, test_gen = get_datasets(ARGS.dsname, ARGS.regions, ("val", "test"))

    # train set evaluation
    # if "train" in ARGS.eval_sets:
        # raise NotImplementedError()
        # train_gen = datasets["train"]
        # train_gen.training = False
        # train_gen.summary()
        # if not ARGS.nolosses:
        #     evaluate_loss_metrics(train_gen, model, MODEL_DIR)
        # evaluate_pointmatching(train_gen, model, MODEL_DIR)

    # validation set evaluation
    if "val" in ARGS.eval_sets:
        val_gen.summary()
        val_dir = MODEL_DIR.joinpath("results_validation")
        if not ARGS.nolosses:
            evaluate_loss_metrics(val_gen, model, val_dir)
        preds_val, X_val = generate_predictions(val_gen, model, val_dir)

        # estimate best params on validation set
        params, gridparams = estimate_postproc_params(preds_val, val_gen.gt_full, val_gen.bounds_full, val_dir)
        # evaluate
        evaluate_postproc_params(val_gen, val_dir, preds_val, X_val, 
                params=params, gridparams=gridparams)

    if "test" in ARGS.eval_sets:
        # load best params estimated on val set from file
        val_path = MODEL_DIR.joinpath("results_validation", "results_pointmatch.json")
        if not os.path.exists(val_path):
            raise FileNotFoundError("Validation pointmatch results file must exist to run eval on the test set. Supply 'val' as an arg to the --eval-sets command-line parameter")
        with open(val_path) as f:
            val_results = json.load(f)
        params, gridparams = val_results["params"], val_results["gridparams"]

        # test set evaluation
        test_gen.summary()
        test_dir = MODEL_DIR.joinpath("results_test")
        if not ARGS.nolosses:
            evaluate_loss_metrics(test_gen, model, test_dir)
        preds_test, X_test = generate_predictions(test_gen, model, test_dir)

        # use val-set estimated params on test set
        evaluate_postproc_params(test_gen, test_dir, preds_test, X_test, 
                params=params, gridparams=gridparams)

    timer.measure("done evaluating " + ARGS.name)


if __name__ == "__main__":
    """
    Parse arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--name",required=True,help="name of model to run, with possible timestamp in front")
    parser.add_argument("--sets",dest="eval_sets",default=("val","test"),nargs="+",choices=("val", "test"),
                help="datasets to evaluate on: train, val, and/or test. test automatically selects val as well")
    parser.add_argument("--noplot",action="store_true",help="don't make plots (significantly improves speed)")
    parser.add_argument("--nolosses",action="store_true",help="do not compute loss metrics")
    parser.add_argument("--save-preds",action="store_true",help="save raw predictions as npy files")
    parser.add_argument("--test",action="store_true",help="just do a test run to make sure imports and such are working")
    parser.parse_args(namespace=ARGS)

    main()
