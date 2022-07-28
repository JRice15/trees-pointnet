"""
utilities related to visualization & plotting
"""

import contextlib
import glob
import os
import json
import time
from pathlib import PurePath

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors
import numpy as np

from src import ARGS, DATA_DIR
from common.visualization import plot_markers, plot_NAIP, plot_raster, make_marker_dict


def height_1_gaussian(x, center, sigma):
    """
    gaussian scaled to always have a height of 1
    see for height of gaussian https://stats.stackexchange.com/questions/143631/height-of-a-normal-distribution-curve
    args:
        x: locations to evaluate gassian curve at, shape (X,Y,ncoords)
        center: peak of gaussian
        sigma: std.dev of gaussian kernel
    returns:
        grid, shape (X,Y)
    """
    # constant scalings
    ### these are the real values, but when multiplied together cancels out
    ### height_scale = sigma * np.sqrt(2 * np.pi)
    ### prefix = 1 / (sigma * np.sqrt(2 * np.pi))

    # distances for each point in x
    squared_dist = np.sum((x - center) ** 2, axis=-1)
    # gaussian formula
    exp_factor = -1 / (2 * sigma ** 2)
    return np.exp( exp_factor * squared_dist )


def rasterize_pts_gaussian_blur(bounds, pts, weights, abs_sigma=None, rel_sigma=None, 
        mode="sum", resolution=64):
    """
    rasterize weighted points to a grid by gaussian blurring
    args:
        bounds: Bounds object
        pts: (x,y) locations within the bounds
        weights: corresponding weights, (negative weights will be set to zero)
        mode: how to aggregate values at each grid location. 
            can be a string, or list of strings
            options: max, sum, second-highest, none
        {rel|abs}_sigma: specify relative (fraction of side length) or absolute distance sigma of gaussian smoothing kernel
        resolution: size length of grid
    returns:
        gridvals: N x M grid of weighted values, except if mode is a list gridvals will be a dict mapping mode str to these grids
        gridpts: N x M x 2 grid, representing the x,y coordinates of each pixel
    """
    xmin, xmax, ymin, ymax = bounds.xy_fmt()

    # get gaussian kernel std.dev.
    assert (rel_sigma is None) != (abs_sigma is None) # only one can and must be true
    if rel_sigma is not None:
        size_factor = (xmax-xmin + ymax-ymin) / 2 # mean side length of raster
        gaussian_sigma = rel_sigma * size_factor
    else:
        gaussian_sigma = abs_sigma

    x = np.linspace(xmin, xmax, resolution)
    y = np.linspace(ymin, ymax, resolution)
    x, y = np.meshgrid(x, y)
    gridpts = np.stack([x,y], axis=-1)

    # initialize data for all modes
    gridvals = {
        "second-highest": [],
        "sum": np.zeros_like(x),
        "max": np.zeros_like(x),
    }

    # make mode always be a list of str
    if isinstance(mode, str):
        mode_list = [mode]
    else:
        assert isinstance(mode, list)
        mode_list = mode
        assert len(mode_list)
    for m in mode_list:
        assert m in gridvals.keys()

    # filter out small weights: 
    # 1/100th of max weight or 1e-3, whichever is smaller
    min_weight = min(0.01 * max(weights), 1e-3)
    mask = (weights > min_weight)
    pts = pts[mask]
    weights = weights[mask]

    for point,weight in zip(pts, weights):
        new_vals = weight * height_1_gaussian(gridpts, point, sigma=gaussian_sigma)

        if "sum" in mode_list:
            gridvals["sum"] += new_vals
        if "max" in mode_list:
            gridvals["max"] = np.maximum(gridvals["max"], new_vals)
        if "second-highest" in mode_list:
            # collect all values
            gridvals["second-highest"].append(new_vals)
    
    if "second-highest" in mode_list:
        second_highest = gridvals["second-highest"]
        second_highest = np.stack(second_highest, axis=0)
        # get second largest along channels dim
        gridvals["second-highest"] = np.sort(second_highest, axis=0)[-2]

    if not isinstance(mode, list):
        gridvals = gridvals[mode]
    else:
        gridvals = {k:v for k,v in gridvals.items() if k in mode_list}
    return gridvals, gridpts



def rasterize_pts_pixelwise(bounds, pts, weights, mode="sum", resolution=64,
        empty_value=0.0):
    """
    rasterize weighted points to a grid based on which pixel they fall in (no blurring/smoothing)
    args:
        bounds: Bounds object
        pts: (x,y) locations within the bounds
        weights: corresponding weights, (negative weights will be set to zero)
        mode: str, how to aggregate values at each grid location. options: max, sum, second-highest, absmax
        resolution: side length of grid
    returns:
        gridvals: N x M grid of weighted values
        gridpts: N x M x 2 grid, representing the x,y coordinates of each pixel
    """
    xmin, xmax, ymin, ymax = bounds.xy_fmt()

    x_ticks = np.linspace(xmin, xmax, resolution)
    y_ticks = np.linspace(ymin, ymax, resolution)
    x_grid, y_grid = np.meshgrid(x_ticks, y_ticks)
    gridpts = np.stack([x_grid, y_grid], axis=-1)

    # get stepsizes
    x_step = np.mean(np.diff(x_ticks))
    y_step = np.mean(np.diff(y_ticks))

    # initialize grid for aggregation methods
    if mode == "second-highest":
        gridvals_highest = np.full_like(x_grid, empty_value)
        gridvals_secondhighest = np.full_like(x_grid, empty_value)
    else:
        gridvals = np.full_like(x_grid, empty_value)

    x_locs = pts[:,0]
    y_locs = pts[:,1]
    x_pixels = ((x_locs - xmin) // x_step).astype(int)
    y_pixels = ((y_locs - ymin) // y_step).astype(int)

    for x,y,weight in zip(x_pixels, y_pixels, weights):
        if mode == "sum":
            gridvals[y,x] += weight
        elif mode == "max":
            gridvals[y,x] = max(gridvals[y,x], weight)
        elif mode == "absmax":
            if abs(weight) > abs(gridvals[y,x]):
                gridvals[y,x] = weight
        elif mode == "second-highest":
            # collect all values
            if weight > gridvals_highest[y,x]:
                gridvals_secondhighest[y,x] = gridvals_highest[y,x]
                gridvals_highest[y,x] = weight
            elif weight > gridvals_secondhighest[y,x]:
                gridvals_secondhighest[y,x] = weight
        else:
            raise ValueError("Unknown rasterize_pts mode")
    
    if mode == "second-highest":
        gridvals = gridvals_highest

    return gridvals, gridpts


def rasterize_and_plot(pts, bounds, filename, *, gaussian_blur=True, rel_sigma=None, abs_sigma=None, weights=None, 
        title=None, clip=None, sqrt_scale=False, mode="sum", markers=None, 
        weight_label=None, grid_resolution=None, colorbar=True, ticks=False,
        empty_value=0.0):
    """
    create raster plot of points, with optional weights
    args:
        gaussian_blur: whether to rasterize with gaussian blur, or directly pixelwise
        guassian_sigma: stddev of kernel to use when raster smoothing, where 1 is the width of the plot
        clip: None, or max value for output
        mode: max, or sum. method of generating y values in raster
        {rel|abs}_sigma: specify relative (fraction of side length) or absolute distance sigma of gaussian smoothing kernel
        mark: dict mapping names to points to mark with an x, array shape (n,2)
    """
    if gaussian_blur:
        gridvals, gridcoords = rasterize_pts_gaussian_blur(bounds, pts, weights, 
                                rel_sigma=rel_sigma, abs_sigma=abs_sigma, 
                                mode=mode, resolution=grid_resolution)
    else:
        gridvals, gridcoords = rasterize_pts_pixelwise(bounds, pts, weights, 
                                mode=mode, resolution=grid_resolution,
                                empty_value=empty_value)

    if sqrt_scale:
        gridvals = np.sqrt(gridvals)
    if clip is not None:
        gridvals = np.clip(gridvals, None, clip)

    plot_raster(gridvals, 
        gridcoords=gridcoords, 
        filename=filename, 
        colorbar_label=weight_label,
        markers=markers,
        title=title,
        colorbar=colorbar,
        ticks=ticks)




def plot_one_example(outdir, patch_id, *, Y, bounds, X=None, pred=None, pred_peaks=None, 
        naip=None, grid_resolution=128, pointmatch_inds=None):
    """
    generate and save raster plots for one example input and output from a dataset
    args:
        outdir: pathlib.PurePath
        patch_id
        Y: targets from patch generator
        X: input from patch generator
        bounds
        pred: raw predictions from network
        pred_peaks: the true final predictions
        naip: naip image
    returns: none
    """
    if not os.path.exists(outdir):
        os.makedirs(outdir, exist_ok=True)

    patchname = "_".join([str(i) for i in patch_id])

    if Y.shape[-1] > 2:
        ylocs = Y[Y[...,2] > 0.5][...,:2]
    else:
        ylocs = Y

    gt_ntrees = len(ylocs)

    markers = make_marker_dict(gt=ylocs, preds=pred_peaks, pointmatch_inds=pointmatch_inds)

    if X is not None:
        x_locs = X[...,:2]

        # lidar height (second-highest mode, to reduce noise)
        x_heights = X[...,2]
        rasterize_and_plot(
            x_locs, 
            weights=x_heights, 
            weight_label="height",
            gaussian_blur=False,
            mode="second-highest", 
            filename=outdir.joinpath("{}_lidar_height".format(patchname)), 
            markers=markers, 
            bounds=bounds,
            grid_resolution=grid_resolution)
        
        # lidar ndvi
        x_ndvi = X[...,-1]
        rasterize_and_plot(
            x_locs, 
            weights=x_ndvi, 
            weight_label="ndvi", 
            mode="max",
            gaussian_blur=False,
            filename=outdir.joinpath("{}_lidar_ndvi".format(patchname)),
            markers=markers, 
            bounds=bounds,
            grid_resolution=grid_resolution,
            empty_value=0.0)

    if pred is not None:
        # prediction confidence raster
        pred_locs = pred[...,:2]
        pred_weights = pred[...,2]
        rasterize_and_plot(
            pred_locs, 
            weights=pred_weights, 
            weight_label="prediction confidence (per-pixel max)",
            gaussian_blur=False,
            filename=outdir.joinpath("{}_pred_raw".format(patchname)),
            mode="max", 
            markers=markers, 
            bounds=bounds,
            grid_resolution=grid_resolution)


    if naip is not None:
        plot_NAIP(
            naip=naip, 
            bounds=bounds,
            filename=outdir.joinpath(patchname+"_NAIP_RGB"),
            markers=markers,
        )





