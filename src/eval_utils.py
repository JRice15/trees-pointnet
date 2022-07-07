import logging
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
import ray
import tensorflow as tf
from scipy.optimize import linear_sum_assignment
from skimage.feature import peak_local_max
from sklearn.metrics import pairwise_distances
from tensorflow import keras
from tensorflow.keras import backend as K
from tqdm import tqdm

from src import ARGS, DATA_DIR, MODEL_SAVE_FMT, REPO_ROOT
from src.utils import (glob_modeldir, rasterize_pts_gaussian_blur, group_by_composite_key,
                       plot_one_example, rasterize_and_plot, scaled_0_1, load_params_into_ARGS)

# max dist that two points can be considered matched, in meters
MAX_MATCH_DIST = 6
# size of grids to rasterize
GRID_RESOLUTION = 128


"""
Pointmatching + utils
"""

def pointmatch(all_gts, all_preds, prune_unpromising=True):
    """
    args:
        all_gts: dict, mapping patchid to array of shape (ntrees,2) where channels are (x,y)
        all_preds: dict, mappin patchid to array of shape (npatches,npoints,3) where channels are (x,y,confidence)
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

    for i,patch_id in enumerate(all_gts.keys()):
        # pruning early if we've gotten only 0 or 1 true positives
        if i == 50 and prune_unpromising:
            if all_tp <= 1:
                pruned = True
                break

        gt = all_gts[patch_id]
        # get pred, or empty array if missing from dict
        try:
            pred = all_preds[patch_id]
        except KeyError:
            pred = np.empty((0,3))

        if len(gt) == 0:
            all_fp += len(pred)
            continue
        elif len(pred) == 0:
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
        # tp_inds = np.where(np.any(np.logical_not(np.isinf(dists)),axis=0))[0]
        tp_inds = np.where(~np.isinf(dists))[1]
        all_tp += len(tp_inds)

        # un-associated pred trees = false positives
        fp_inds = np.where(np.all(np.isinf(dists),axis=0))[0]
        all_fp += len(fp_inds)

        # un-associated gt trees = false negatives
        fn_inds = np.where(np.all(np.isinf(dists),axis=1))[0]
        all_fn += len(fn_inds)
        
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
    }
    return results



def find_local_maxima(pred_grids, pred_coords, min_dist=1, conf_threshold=None):
    """
    args:
        pred_grid: dict mapping patchid to raster, shape (N,N)
        pred_coords: dict mapping patchid to coordinates of each raster pixel, shape (N,N,2)
        conf_threshold: smallest conf that a peak is allowed to have
        min_distance: smallest allowed distance, in NAIP pixels (0.6 meters), between peaks
    returns:
        dict: mapping patch id to local peaks, shape (N,3) where channels are x,y,confidence
    """
    maxima = {}
    for patch_id, gridvals in pred_grids.items():
        gridcoords = pred_coords[patch_id]

        # get average pixel width
        n_pixels = gridvals.shape[0]
        min_dist_pixels = round(min_dist * (256 / n_pixels))

        # find local maxima
        peak_pixel_inds = peak_local_max(gridvals, 
                                threshold_abs=conf_threshold, # must meet the min threshold
                                min_distance=min_dist_pixels)
        peak_pixel_inds = tuple(peak_pixel_inds.T)

        # get geo coords of the maxima pixels
        peak_confs = gridvals[peak_pixel_inds]
        peak_coords = gridcoords[peak_pixel_inds]
        # convert back to format of input preds
        peak_pts = np.concatenate((peak_coords, peak_confs[...,np.newaxis]), axis=-1)
        maxima[patch_id] = peak_pts

    return maxima


@ray.remote
def _ray_rasterize_pts_blur(*args, **kwargs):
    """
    distribute rasterization using Ray
    """
    return rasterize_pts_gaussian_blur(*args, **kwargs)

def rasterize_preds(preds, bounds, grid_aggs, is_subdiv=False):
    """
    args:
        preds: dict mapping patch id to pred pts (must be original CRS, not 0-1 scale)
        bounds: dict mapping patch id to Bounds
        modes: list of grid agg modes
        is_subdiv: bool, whether patches are subdiv or full-sized
    returns:
        pred_grids: dict mapping grid_agg mode to dict mapping p_id to raster, shape (N,N)
        pred_coords: dict mapping p_id to coords of raster, shape (N,N,2)
    """
    if is_subdiv:
        resolution = GRID_RESOLUTION // ARGS.subdivide
    else:
        resolution = GRID_RESOLUTION

    # initialize ray, silence output
    if not ray.is_initialized():
        ray.init(
            log_to_driver=False
        )

    futures = []
    patch_ids = []
    for p_id, pred in preds.items():
        future = _ray_rasterize_pts_blur.remote(
                bounds[p_id], pred[:,:2], pred[:,2], 
                abs_sigma=ARGS.gaussian_sigma, mode=grid_aggs,
                resolution=resolution)
        futures.append(future)
        patch_ids.append(p_id)
    
    results = ray.get(futures)
    pred_grids = {mode: {} for mode in grid_aggs}
    pred_coords = {}
    for p_id, (vals_dict, coords) in zip(patch_ids, results):
        for (mode, vals) in vals_dict.items():
            pred_grids[mode][p_id] = vals
        pred_coords[p_id] = coords

    return pred_grids, pred_coords


"""
Overlap methods
"""

def drop_overlaps(X, bounds_subdiv=None):
    """
    args:
        X: some dict, mapping patch subdiv id to pts
    returns:    
        X, with overlap patches dropped, grouped and concatenated by full patch id
    """
    # drop overlap patches
    X = {key: pts for key, pts in X.items() if key[-2] % 2 == 0 and key[-1] % 2 == 0}
    # group and concat
    return group_by_composite_key(X, first_n=2,
            agg_f=lambda x: np.concatenate(list(x.values()), axis=0)
        )


def overlap_by_mid_cutoff(preds_subdiv, bounds_subdiv):
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
        xmin, ymin, xmax, ymax = bounds_subdiv[patch_id].minmax_fmt()
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
                        agg_f=lambda x: np.concatenate(list(x.values()), axis=0))

    return preds_combined


OVERLAP_METHODS = {
    "drop": drop_overlaps,
    "cut": overlap_by_mid_cutoff,
}


"""
Visualization
"""

def viz_predictions(patchgen, outdir, *, X_subdiv, X_full, Y_full, Y_subdiv, 
        preds_subdiv, preds_full, pred_grids, pred_peaks):
    """
    data visualizations
    """
    print("Generating visualizations...")
    os.makedirs(outdir, exist_ok=True)

    patch_ids = sorted(preds_full.keys())
    n_ids = len(patch_ids)
    subpatch_ids = sorted(preds_subdiv.keys())

    # grab random 10ish examples
    step_size = max(1, n_ids//10)
    for p_id in tqdm(patch_ids[::step_size]):
        patch_name = "_".join([str(x) for x in p_id])
        patch_dir = outdir.joinpath(patch_name)
        # first three subpatch ids
        these_subpatch_ids = [key for key in subpatch_ids if key[:2] == p_id]
        for subp_id in these_subpatch_ids[:3]:
            subpatch_dir = patch_dir.joinpath("subpatches")

            bounds = patchgen.get_patch_bounds(subp_id)
            plot_one_example(
                outdir=subpatch_dir,
                patch_id=subp_id,
                X=X_subdiv[subp_id],
                Y=Y_subdiv[subp_id],
                pred=preds_subdiv[subp_id],
                # pred_peaks=bounds.filter_pts(pred_peaks[p_id]), # peaks within this subpatch
                naip=patchgen.get_naip(subp_id),
                grid_resolution=GRID_RESOLUTION//ARGS.subdivide,
                zero_one_bounds=True
            )

        # plot full-patch data
        naip = patchgen.get_naip(p_id)
        plot_one_example(
            outdir=patch_dir,
            patch_id=p_id, 
            X=X_full[p_id], 
            Y=Y_full[p_id], 
            pred=preds_full[p_id], 
            pred_overlap_gridded=pred_grids[p_id],
            pred_peaks=pred_peaks[p_id],
            naip=patchgen.get_naip(p_id), 
            grid_resolution=GRID_RESOLUTION,
        )

