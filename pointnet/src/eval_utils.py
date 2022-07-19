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

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import ray
import tensorflow as tf
from skimage.feature import peak_local_max
from tensorflow import keras
from tensorflow.keras import backend as K
from tqdm import tqdm

from src import ARGS, DATA_DIR
from src.utils import (glob_modeldir, group_by_composite_key)
from src.viz_utils import (plot_one_example, rasterize_pts_gaussian_blur)

from common.data_handling import Bounds
from common.pointmatch import pointmatch


# size of grids to rasterize
GRID_RESOLUTION = 128


"""
Pointmatching utils
"""


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
    drop "odd" patches, so that the remaining even ones do not overlap
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


def overlap_by_mid_buffer(preds_subdiv, bounds_subdiv):
    """
    overlaps sub-patches back into full patches by only keeping the middle 50% of 
    each subpatch (except for subpatched on the edges)
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
    "buffer": overlap_by_mid_buffer,
}


def viz_predictions(patchgen, outdir, *, X_subdiv, Y_full, Y_subdiv, 
        preds_full, preds_full_peaks, bounds_full, pointmatch_inds=None,
        preds_subdiv=None):
    """
    data visualizations
    args:
        *_subdiv: subdivided, normalize versions of data
        *_full: full (overlapped), denormalize (original CRS) versions of data
        preds_full_peaks: post-processed version of raw preds that has final predicted locations
        pointmatch_inds: dict with keys tp,fp,fn that has
    """
    print("Denormalizing & overlapping X...")
    # denormalize X
    X_subdiv_unnormed = {p_id: patchgen.denormalize_pts(pts, p_id) for p_id,pts in X_subdiv.items()}

    # drop overlapping subpatches, combine into full patches
    X_full = drop_overlaps(X_subdiv_unnormed)
    del X_subdiv_unnormed

    print("Generating visualizations...")
    os.makedirs(outdir, exist_ok=True)

    patch_ids = sorted(preds_full.keys())
    n_ids = len(patch_ids)
    subpatch_ids = sorted(X_subdiv.keys())

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
                pred=preds_subdiv[subp_id] if preds_subdiv is not None else None,
                # pred_peaks=bounds.filter_pts(pred_peaks[p_id]), # peaks within this subpatch
                naip=patchgen.get_naip(subp_id),
                grid_resolution=GRID_RESOLUTION//ARGS.subdivide,
                bounds=Bounds.zero_to_one(),
            )

        # plot full-patch data
        plot_one_example(
            outdir=patch_dir,
            patch_id=p_id, 
            X=X_full[p_id], 
            Y=Y_full[p_id], 
            pred=preds_full[p_id],
            pred_peaks=preds_full_peaks[p_id] if p_id in preds_full_peaks else None,
            naip=patchgen.get_naip(p_id), 
            grid_resolution=GRID_RESOLUTION,
            bounds=bounds_full[p_id],
            pointmatch_inds=pointmatch_inds[p_id],
        )
    


