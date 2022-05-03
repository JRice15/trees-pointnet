
import argparse
import contextlib
import datetime
import glob
import json
import os
import time
from pathlib import PurePath
from pprint import pprint

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from scipy.optimize import linear_sum_assignment
from skimage.feature import peak_local_max
from sklearn.metrics import pairwise_distances
from tensorflow import keras
from tensorflow.keras import backend as K
from tqdm import tqdm

from src import ARGS, DATA_DIR, MODEL_SAVE_FMT, REPO_ROOT
from src.patch_generator import get_datasets
from src.losses import get_loss
from src.models import pointnet
from src.tf_utils import load_saved_model
from src.utils import (glob_modeldir, gridify_pts, group_by_composite_key,
                       plot_one_example, raster_plot, scaled_0_1)

# max dist that two points can be considered matched, in meters
MAX_MATCH_DIST = 6
# size of grids to gridify
GRID_RESOLUTION = 256



def pointmatch(all_gts, all_preds, conf_threshold, prune_unpromising=True):
    """
    args:
        all_gts: array of shape (npatches,npoints,3) where channels are (x,y,isvalid)
        all_preds: array of shape (npatches,npoints,3) where channels are (x,y,confidence)
        conf_threshold: abs confidence threshold that predicted points must meet to be considered a predicted tree
        prune_unpromising: whether to stop if 1 or fewer true positives are recorded after 100 patches
    returns:
        dict: precision, recall, fscore, rmse, pruned (bool)
    """
    COST_MATRIX_MAXVAL = 1e10

    all_tp = 0
    all_fp = 0
    all_fn = 0
    all_tp_dists = []
    pruned = False

    for i in range(len(all_gts)):
        # pruning early
        if i == 100 and prune_unpromising:
            if all_tp <= 1:
                pruned = True
                break

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
        
        # trees must be within max match distance
        dists[dists>MAX_MATCH_DIST] = np.inf

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
    
    # calling float/int on a lot of these because json doesn't like numpy floats/ints
    results = {
        'pruned': pruned,
        'tp': int(all_tp),
        'fp': int(all_fp),
        'fn': int(all_fn),
        'precision': float(precision),
        'recall': float(recall),
        'fscore': float(fscore),
        'rmse': float(rmse),
        'threshold': float(conf_threshold),
    }
    return results



def find_local_maxima(pred_grids, bounds, min_conf_threshold=None):
    """
    args:
        pred_grids: outputs of gridify_preds(...)
        bounds: list of Bounds
        min_conf_threshold: smallest conf threshold
    returns:
        dict: mapping patch id to local peaks, shape (N,3) where channels are x,y,confidence
    """
    maxima = {}
    for patch_id,pred_dict in tqdm(pred_grids.items(), total=len(pred_grids)):
        gridvals = pred_dict["vals"]
        gridcoords = pred_dict["coords"]

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
        maxima[patch_id] = peak_pts

    return maxima


def gridify_preds(preds, bounds):
    """
    args:
        preds: dict mapping patch id to pred pts (must be original CRS, not 0-1 scale)
        bounds: dict mapping patch id to Bounds
    returns:
        dict: mapping patch id to another dict, with keys "vals" and "coords"
    """
    pred_grids = {}
    for key, pred in preds.items():
        vals, coords = gridify_pts(bounds[key], pred[:,:2], pred[:,2], 
                abs_sigma=ARGS.gaussian_sigma, mode=ARGS.grid_agg) # TODO test both grid-agg modes?
        pred_grids[key] = {"vals": vals, "coords": coords}

    # pred_grids_grouped = group_by_composite_key(pred_grids, first_n=2)
    return pred_grids


def overlap_by_hard_cutoff(preds_subdiv, bounds_subdiv):
    """
    converts overlapping sub-patches back into full patches
    args:
        preds_subdiv: dict, mapping subdiv patch id to pts
        bounds_subdiv: dict, mappng subdiv patch id to bounds
    returns:
        dict: mapping full patch id to pts
    """
    preds_subdiv_chopped = {}
    for patch_id, preds in preds_subdiv.items():
        xmin, ymin, xmax, ymax = bounds_subdiv[patch_id].to_minmax()
        x_cut = (xmax - xmin) / 4
        y_cut = (ymax - ymin) / 4
        region, patchnum, px, py = patch_id
        # check for neighbor patches
        if (region, patchnum, px-1, py) in preds_subdiv:
            xmin += x_cut
        if (region, patchnum, px+1, py) in preds_subdiv:
            xmax -= x_cut
        if (region, patchnum, px, py-1) in preds_subdiv:
            ymin += y_cut
        if (region, patchnum, px, py+1) in preds_subdiv:
            ymax -= y_cut
        # make relevant chops
        x = preds[:,0]
        y = preds[:,1]
        preds = preds[(x >= xmin) & (x < xmax) & (y >= ymin) & (y < ymax)]
        preds_subdiv_chopped[patch_id] = preds

    # group by (region, patchnum) and concat
    preds_combined = group_by_composite_key(preds_subdiv_chopped, first_n=2,
                        agg_f=lambda x: np.concatenate(list(x.values(), axis=0)))

    return preds_combined




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

    """
    generate predictions
    """
    print("\nGenerating predictions...")
    assert patchgen.batch_size == 1
    
    X_subdiv, _ = patchgen.load_all()
    X_subdiv = np.squeeze(X_subdiv.numpy())
    preds_subdiv = np.squeeze(model.predict(X_subdiv))

    # associate each pred set with its patch id
    preds_subdiv = dict(zip(patchgen.patch_ids, preds_subdiv))

    # denormalize data
    for patch_id,pts in preds_subdiv.items():
        pts[:,:2] = patchgen.denormalize_pts(pts[:,:2], patch_id)
        preds_subdiv[patch_id] = pts

    # combine with overlap
    preds_full = overlap_by_hard_cutoff(preds_subdiv, patchgen.bounds_subdiv)

    # find localmax peak predictions
    print("Finding prediction maxima...")
    pred_peaks = find_local_maxima(preds_full, patchgen.bounds_full, min_conf_threshold=min(pointmatch_thresholds))

    """
    Pointmatching
    """
    print("Pointmatching...")
    # get full ground-truth
    Y = patchgen.gt_full

    results = []
    for thresh in tqdm(pointmatch_thresholds):
        result = pointmatch(Y, pred_peaks, thresh)
        results.append(result)

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
    with open(outdir.joinpath("results_pointmatch.json"), "w") as f:
        json.dump(pointmatch_stats, f, indent=2)

    return pointmatch_stats


def evaluate_loss_metrics(patchgen, model, outdir):
    """
    Evaluate model's builtin metrics on a dataset
    args:
        patchgen
    """
    print("Evaluating model's metrics...")

    metric_vals = model.evaluate(patchgen)
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

    # load original params into ARGS object
    params_file = MODEL_DIR.joinpath("params.json")
    with open(params_file, "r") as f:
        params = json.load(f)
    params.pop("name")
    params.pop("test")
    params.pop("noplot")
    for k,v in params.items():
        setattr(ARGS, k, v)
    ARGS.test = False

    pprint(vars(ARGS))

    """
    Evaluation
    """

    model = load_saved_model(MODEL_DIR)

    datasets = get_datasets(ARGS.dsname, ARGS.regions, ARGS.eval_sets, train_batchsize=1, val_batchsize=1)
    datasets = dict(zip(ARGS.eval_sets, datasets))

    # grid search approximating log-scale
    THRESHOLDS = list(np.arange(0, 0.030, 0.005)) + list(np.arange(0.030, 0.100, 0.010)) + list(np.arange(0.100, 1, 0.025))
    # GRIDIFY_MODES = ["sum", "max"]

    # train set evaluation
    if "train" in ARGS.eval_sets:
        train_gen = datasets["train"]
        train_gen.training = False
        train_gen.summary()
        evaluate_loss_metrics(train_gen, model, MODEL_DIR)
        evaluate_pointmatching(train_gen, model, MODEL_DIR, THRESHOLDS)

    # validation set evaluation
    if "val" in ARGS.eval_sets or "test" in ARGS.eval_sets:
        val_gen = datasets["val"]
        val_gen.summary()
        evaluate_loss_metrics(val_gen, model, MODEL_DIR)
        pointmatch_stats = evaluate_pointmatching(val_gen, model, MODEL_DIR, THRESHOLDS)
        val_best_thresh = pointmatch_stats["best"]["threshold"]

    # test set evaluation
    if "test" in ARGS.eval_sets:
        test_gen = datasets["test"]
        test_gen.summary()
        evaluate_loss_metrics(test_gen, model, MODEL_DIR)
        evaluate_pointmatching(test_gen, model, MODEL_DIR, [val_best_thresh])




if __name__ == "__main__":
    """
    Parse arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--name",required=True,help="name of model to run, with possible timestamp in front")
    parser.add_argument("--sets",dest="eval_sets",default=("train","val","test"),nargs="+",help="datasets to evaluate on: train, val, and/or test. test automatically selects val as well")
    parser.add_argument("--noplot",action="store_true")
    parser.parse_args(namespace=ARGS)

    main()
