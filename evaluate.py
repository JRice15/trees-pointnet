
import contextlib
import datetime
import glob
import json
import argparse
import os
from pprint import pprint
from pathlib import PurePath
import time

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from sklearn.metrics import pairwise_distances
from scipy.optimize import linear_sum_assignment
from skimage.feature import peak_local_max

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model
from tensorflow.keras import backend as K
from tensorflow.keras import callbacks, layers
from tensorflow.keras.optimizers import Adam

from src import DATA_DIR, REPO_ROOT, ARGS, patch_generator
from src.losses import get_loss
from src.models import pointnet
from src.tf_utils import MyModelCheckpoint, output_model, load_saved_model
from src.utils import raster_plot, glob_modeldir, scaled_0_1, gridify_pts


MAX_MATCH_DIST = 10

def parse_eval_args():
    """
    Parse arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--name",required=True,help="name of model to run, with possible timestamp in front")
    parser.parse_args(namespace=ARGS)


def errors_plot(pred, y, results_dir):
    """error plot for predicted vs gt counts"""
    x_w = np.empty(pred.shape)
    x_w.fill(1/pred.shape[0])
    y_w = np.empty(y.shape)
    y_w.fill(1/y.shape[0])
    low = int(min(pred.min(), y.min()))
    high = int(max(pred.max(), y.max()))
    step = max((high - low) // 20, 1)
    bins = range(low, high+1, step)
    plt.hist(y, bins=bins, weights=y_w, label="gt", alpha=0.5, color="green")
    plt.hist(pred, bins=bins, weights=x_w, label="predictions", alpha=0.5, color="blue")
    plt.title("Predictions and Ground Truth Values")
    plt.axvline(np.mean(y), label="gt mean", color="green", linestyle="--")
    plt.axvline(np.mean(pred), label="prediction mean", color="blue", linestyle="--")
    plt.legend()
    plt.savefig(os.path.join(results_dir, "preds_vs_gt_hist.png"))
    plt.close()

def count_errors(pred, y):
    if len(pred.shape) > 1:
        pred = K.sum(pred[...,-1], axis=-1)
        y = K.sum(y[...,-1], axis=-1)
    return pred - y


def plot_one_example(x, y, patch_id, outdir, pred=None, pred_peaks=None, 
        naip=None, has_ndvi=False, zero_one_bounds=False):
    """
    generate raster plots for one example input and output from a dataset
    args:
        x: input from patch generator
        y: targets from patch generator
        patch_id
        outdir: pathlib.PurePath
        pred: raw predictions from network
        pred_peaks: thresholded peaks from the predictions blurred to grid; ie the true final predictions
        naip: naip image
        has_ndvi: bool, whether x has ndvi as last channel
    """
    if not os.path.exists(outdir):
        os.makedirs(outdir, exist_ok=True)

    patchname = "_".join([str(x) for x in patch_id])
    ylocs = y[y[...,2] == 1][...,:2]

    gt_ntrees = len(ylocs)
    x_locs = x[...,:2]
    x_heights = x[...,2]

    if zero_one_bounds:
        sigma = scaled_0_1(ARGS.gaussian_sigma)
    else:
        sigma = ARGS.gaussian_sigma

    markings = {"gt trees": ylocs}
    if pred_peaks is not None:
        markings["predicted trees"] = pred_peaks[...,:2]

    # lidar height (second-highest mode, to avoid noise)
    raster_plot(x_locs, weights=x_heights, 
        weight_label="height",
        abs_sigma=sigma, 
        mode="second-highest", 
        filename=outdir.joinpath("{}_lidar_height".format(patchname)), 
        mark=markings, 
        zero_one_bounds=zero_one_bounds)
    
    # lidar ndvi
    if has_ndvi:
        x_ndvi = x[...,3]
        raster_plot(x_locs, weights=x_ndvi, 
            weight_label="ndvi", 
            mode="max",
            abs_sigma=sigma, 
            filename=outdir.joinpath("{}_lidar_ndvi".format(patchname)),
            mark=markings, 
            zero_one_bounds=zero_one_bounds)

    if pred is not None:
        # prediction confidence raster
        pred_locs = pred[...,:2]
        pred_weights = pred[...,2]
        raster_plot(pred_locs, weights=pred_weights, 
            weight_label="prediction confidence",
            abs_sigma=sigma, 
            filename=outdir.joinpath("{}_pred".format(patchname)),
            mode="sum", 
            mark=markings, 
            zero_one_bounds=zero_one_bounds)

    if naip is not None:
        plt.imshow(naip[...,:3]) # only use RGB
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(outdir.joinpath(patchname+"_NAIP_RGB.png"))
        plt.clf()
        plt.close()


def find_local_maxima(preds, bounds, min_conf_threshold=None, grid_resolution=64):
    """
    args:
        preds: array of shape (npatches,npoints,3) where channels are (x,y,conf)
        bounds: list of Bounds
        conf_threshold
    """
    maxima = []
    for i,pred in enumerate(preds):
        # blur preds to grid
        gridvals, gridcoords = gridify_pts(bounds[i], pred[:,:2], pred[:,2], 
                    abs_sigma=ARGS.gaussian_sigma, resolution=grid_resolution)

        # find local maxima
        peak_pixel_inds = peak_local_max(gridvals, 
                                threshold_abs=min_conf_threshold, # must meet the min threshold
                                min_distance=2) # disallow maxima in adjacent pixels
        peak_pixel_inds = tuple(peak_pixel_inds.T)

        # get geo coords of the maxima pixels
        peak_confs = gridvals[peak_pixel_inds]
        peak_coords = gridcoords[peak_pixel_inds]
        # convert back to format of input preds
        peak_pts = np.concatenate((peak_coords, peak_confs[...,np.newaxis]), axis=-1)
        maxima.append(peak_pts)

    return maxima


def pointmatch(all_gts, all_preds, conf_threshold, max_match_dist):
    """
    args:
        all_gts: array of shape (npatches,npoints,3) where channels are (x,y,isvalid)
        all_preds: array of shape (npatches,npoints,3) where channels are (x,y,confidence)
        conf_threshold: list (must be in sorted order, low to high) of confidence thresholds to test
    returns:
        precisions, recalls, fscores, rmses: each the same length as conf_threshold
    """
    COST_MATRIX_MAXVAL = 1e10

    all_tp = 0
    all_fp = 0
    all_fn = 0
    all_tp_dists = []

    for i in range(len(all_gts)):
        gt = all_gts[i]
        pred = all_preds[i]

        # filter valid gt trees and preds
        gt = gt[gt[:,2] > 0.5]
        pred = pred[pred[:,2] >= conf_threshold]

        if len(gt) == 0:
            all_tp += 0
            all_fp += len(pred)
            all_fn += 0
            continue
        elif len(pred) == 0:
            all_tp += 0
            all_fp += 0
            all_fn += len(gt)
            continue

        # calculate pairwise distances
        dists = pairwise_distances(gt[:,:2],pred[:,:2])
        
        # associate each gt tree with all pred trees within radius
        dists[dists>max_match_dist] = np.inf

        # find optimal assignment
        cost_matrix = np.copy(dists)
        cost_matrix[np.isinf(cost_matrix)] = COST_MATRIX_MAXVAL
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        dists[:] = np.inf
        dists[row_ind,col_ind] = cost_matrix[row_ind,col_ind]
        dists[dists>=COST_MATRIX_MAXVAL] = np.inf
        
        # associated pred trees = true positives
        tp_inds = np.where(np.any(np.logical_not(np.isinf(dists)),axis=0))[0]
        tp = len(tp_inds)

        # un-associated pred trees = false positives
        fp_inds = np.where(np.all(np.isinf(dists),axis=0))[0]
        fp = len(fp_inds)

        # un-associated gt trees = false negatives
        fn_inds = np.where(np.all(np.isinf(dists),axis=1))[0]
        fn = len(fn_inds)
        
        all_tp += tp
        all_fp += fp
        all_fn += fn
        if len(tp_inds):
            tp_dists = np.min(dists[:,tp_inds],axis=0)
            all_tp_dists.append(tp_dists)


    if all_tp + all_fp > 0:
        precision = all_tp/(all_tp+all_fp)
    else:
        precision = 0
    if all_tp + all_fn > 0:
        recall = all_tp/(all_tp+all_fn)
    else:
        recall = 0
    if precision + recall > 0:
        fscore = 2*(precision*recall)/(precision+recall)
    else:
        fscore = 0
    if len(all_tp_dists):
        all_tp_dists = np.concatenate(all_tp_dists)
        rmse = np.sqrt(np.mean(all_tp_dists**2))
    else:
        rmse = -1
    
    results = {
        'tp': int(all_tp),
        'fp': int(all_fp),
        'fn': int(all_fn),
        'precision': float(precision),
        'recall': float(recall),
        'fscore': float(fscore),
        'rmse': float(rmse),
    }
    return results


def run_pointmatching(gts, preds, thresholds):
    """
    args:
        gts: array of shape (npatches,npoints,3) where channels are (x,y,isvalid)
        preds: array of shape (npatches,npoints,3) where channels are (x,y,confidence)
        max_match_dist: max distance (meters) between gt and pred trees that are considered a match
    returns:
        dict of pointmatching stats
    """
    results = []
    for thresh in thresholds:      
        result = pointmatch(gts, preds, thresh, max_match_dist=MAX_MATCH_DIST)
        results.append(result)

    # find best, by fscore
    best_idx = np.argmax([x["fscore"] for x in results])

    best_stats = {k:results[best_idx][k] for k in results[0].keys()}
    best_stats["threshold"] = thresholds[best_idx]

    all_stats = {k:[x[k] for x in results] for k in results[0].keys()}
    all_stats["threshold"] = thresholds

    stats = {
        "best": best_stats,
        "all": all_stats,
    }
    return stats




def evaluate_model(patchgen, model, model_dir, pointmatch_thresholds):
    """
    generate predictions, vizualizations, and evaluation metrics from a model 
    on a LidarPatchGen dataset
    args:
        model: Keras Model to predict with
        model_dir: pathlib.PurePath of model's output
        pointmatch_thresholds: list of floats
    """
    outdir = model_dir.joinpath("results_"+patchgen.name)
    os.makedirs(outdir, exist_ok=True)

    """
    generate predictions
    """
    print("Generating predictions")
    assert patchgen.batch_size == 1
    patchgen.sorted()
    x, y = patchgen.load_all()
    x = np.squeeze(x.numpy())
    y = np.squeeze(y.numpy())
    preds_raw = np.squeeze(model.predict(x))
    patch_ids = patchgen.patch_ids
    patch_bounds = [patchgen.get_patch_bounds(x) for x in patch_ids]

    # denormalize data
    for i in range(len(x)):
        x[i] = patchgen.denormalize_pts(x[i], patch_id=patch_ids[i])
        preds_raw[i,:,:2] = patchgen.denormalize_pts(preds_raw[i,:,:2], patch_id=patch_ids[i])
        y[i,:,:2] = patchgen.denormalize_pts(y[i,:,:2], patch_id=patch_ids[i])

    # find localmax peak predictions
    print("Finding prediction maxima")
    pred_peaks = find_local_maxima(preds_raw, patch_bounds, min_conf_threshold=min(pointmatch_thresholds))

    # save raw sample prediction
    print("Saving raw predictions")
    with open(outdir.joinpath("sample_predictions.txt"), "w") as f:
        f.write("First 5 predictions, ground truths:\n")
        for i in range(min(5, len(preds_raw))):
            f.write("pred raw {}:\n".format(i))
            f.write(str(preds_raw[i])+"\n")
            f.write("pred localmax peaks {}:\n".format(i))
            f.write(str(pred_peaks[i])+"\n")
            f.write("gt {}:\n".format(i))
            f.write(str(y[i])+"\n")
            f.write("first 100 input points {}:\n".format(i))
            f.write(str(x[i,:100])+"\n")

    """
    Pointmatching
    """
    print("Pointmatching")

    pointmatch_stats = run_pointmatching(y, pred_peaks, pointmatch_thresholds)

    print("results:")
    pprint(pointmatch_stats["best"])
    with open(outdir.joinpath("results_pointmatch.json"), "w") as f:
        json.dump(pointmatch_stats, f, indent=2)

    best_thresh = pointmatch_stats["best"]["threshold"]

    # threshold peaks by best threshold
    for i in range(len(pred_peaks)):
        pred_peaks[i] = pred_peaks[i][pred_peaks[i][:,2] >= best_thresh]

    """
    data visualizations
    """
    if ARGS.output_mode in ("seg", "dense"):
        print("Generating visualizations...")
        VIS_DIR = outdir.joinpath("visualizations")
        os.makedirs(VIS_DIR, exist_ok=True)

        # grab random 10 examples
        for i in range(0, len(patch_ids), len(patch_ids)//10):
            naip = patchgen.get_naip(patch_ids[i])
            plot_one_example(x[i], y[i], patch_ids[i], pred=preds_raw[i], naip=naip, 
                has_ndvi=patchgen.use_ndvi, outdir=VIS_DIR, pred_peaks=pred_peaks[i])

    """
    Evaluate Model Metrics
    """
    print("Evaluating model's metrics")

    metric_vals = model.evaluate(patchgen)
    if not isinstance(metric_vals, list):
        results = {"loss": metric_vals}
    else:
        results = {model.metrics_names[i]:v for i,v in enumerate(metric_vals)}

    with open(outdir.joinpath("results_model_metrics.json"), "w") as f:
        json.dump(results, f, indent=2)

    print("\nResults on", patchgen.name, "dataset:")
    for k,v in results.items():
        print(k+":", v)
    
    return pointmatch_stats


def main():
    MODEL_DIR = glob_modeldir(ARGS.name)
    MODEL_PATH = MODEL_DIR.joinpath("model.tf")

    # load original params into ARGS object
    params_file = MODEL_DIR.joinpath("params.json")
    with open(params_file, "r") as f:
        params = json.load(f)
    params.pop("name")
    params.pop("test")
    for k,v in params.items():
        setattr(ARGS, k, v)
    ARGS.test = False

    pprint(vars(ARGS))

    """
    Evaluation
    """

    model = load_saved_model(MODEL_PATH.as_posix())

    train_gen, val_gen = patch_generator.get_train_val_gens(ARGS.dsname, ARGS.regions, train_batchsize=1, val_batchsize=1)

    # train set evaluation
    train_gen.training = False
    train_gen.summary()
    thresholds = list(np.arange(0.025, 1, 0.025))
    pointmatch_stats = evaluate_model(train_gen, model, MODEL_DIR, thresholds)

    # validation set evaluation
    val_gen.summary()
    thresholds = list(np.arange(0.025, 1, 0.025))
    pointmatch_stats = evaluate_model(val_gen, model, MODEL_DIR, thresholds)
    best_thresh = pointmatch_stats["best"]["threshold"]

    # test set evaluation
    test_gen = patch_generator.get_test_gen(ARGS.dsname, ARGS.regions)
    test_gen.summary()
    evaluate_model(test_gen, model, MODEL_DIR, [best_thresh])




if __name__ == "__main__":
    parse_eval_args()
    main()
