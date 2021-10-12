import contextlib
import glob
import os
import time
from pathlib import PurePath

import h5py
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from core import ARGS, DATA_DIR, REPO_ROOT

matplotlib.rc_file_defaults()


def gaussian(x, center, sigma=0.02):
    const = (2 * np.pi * sigma) ** -0.5
    exp = np.exp( -np.sum((x - center) ** 2, axis=-1) / (2 * sigma ** 2))
    return const * exp

def gridify_pts(bounds, pts, weights, gaussian_sigma, mode="sum", resolution=50):
    """
    rasterize weighted points to a grid
    args:
        bounds: (xmin, xmax, ymin, ymax) bounds
        pts: (x,y) locations within the bounds
        weights: corresponding weights, (negative weights will be set to zero)
    returns:
        gridvals: N x M grid of weighted values
        gridpts: N x M x 2 grid, representing the x,y coordinates of each pixel
    """
    xmin, xmax, ymin, ymax = bounds
    x = np.linspace(xmin, xmax, resolution)
    y = np.linspace(ymin, ymax, resolution)
    size_factor = (xmax-xmin + ymax-ymin) / 2 # mean side length of raster
    x, y = np.meshgrid(x, y)
    gridpts = np.stack([x,y], axis=-1)
    gridvals = np.zeros_like(x)
    for i,p in enumerate(pts):
        vals = gaussian(gridpts, p, sigma=gaussian_sigma*size_factor)
        if weights is not None:
            vals *= max(weights[i], 0)
        if mode == "sum":
            gridvals += vals
        elif mode == "max":
            gridvals = np.maximum(gridvals, vals)
        else:
            raise ValueError("Unknown gridify_pts mode")
    return gridvals, gridpts


def raster_plot(pts, filename, gaussian_sigma, weights=None, title=None, clip=None, 
        sqrt_scale=False, mode="sum", mark=None, zero_one_bounds=False):
    """
    create raster plot of points, with optional weights
    args:
        guassian_sigma: stddev of kernel to use when raster smoothing, where 1 is the width of the plot
        clip: None, or max value for output
        mode: max, or sum. method of generating y values in raster
        mark: points to mark with a green x, of shape (n,2) where n is number of points
    """
    if zero_one_bounds:
        xmin, xmax = 0, 1
        ymin, ymax = 0, 1
    else:
        xmin, xmax = pts[:,0].min(), pts[:,0].max()
        ymin, ymax = pts[:,1].min(), pts[:,1].max()

    gridvals, gridpts = gridify_pts([xmin, xmax, ymin, ymax], pts, weights, 
                            gaussian_sigma=gaussian_sigma, mode=mode)

    if sqrt_scale:
        gridvals = np.sqrt(gridvals)
    if clip is not None:
        gridvals = np.clip(gridvals, None, clip)

    x = gridpts[...,0]
    y = gridpts[...,1]
    plt.pcolormesh(x,y,gridvals, shading="auto")
    plt.colorbar()

    if mark is not None:
        plt.scatter(mark[:,0], mark[:,1], c="white", marker="x")

    if title is not None:
        plt.title(title)
    plt.tight_layout()
    plt.savefig(filename)
    plt.clf()
    plt.close()


def glob_modeldir(modelname):
    allmodels_dir = REPO_ROOT.joinpath("models/")

    matching_models = glob.glob(os.path.join(allmodels_dir.as_posix(), modelname+"*", "model.tf"))
    if len(matching_models) > 1:
        print("Multiple models match 'name' argument:")
        print(" ", matching_models)
        print("Defaulting to the most recent:")
        # all the names begin with a date/time string, so sorting gives order by time
        matching_models.sort()
        model_path = PurePath(matching_models[-1])
    elif len(matching_models) == 0:
        print("No matching models!")
        exit()
    else:
        model_path = PurePath(matching_models[0])

    model_dir = model_path.parent
    print(" ", model_dir)

    return model_dir

def get_dataset_dir(dsname=None):
    """
    get the dataset directory from name, or automatically select the one existing dataset if only one exists
    returns:
        dataset_dir, dsname
    """
    if dsname is None:
        existing_datasets = [i for i in os.listdir(DATA_DIR.joinpath("generated")) 
                            if os.path.isdir(DATA_DIR.joinpath("generated").joinpath(i))]
        if len(existing_datasets) > 1:
            raise ValueError("Multiple datasets exist in `data/generated`. Specify which with the --dsname argument")
        elif len(existing_datasets) == 0:
            raise ValueError("No dataset exists in `data/generated`")
        dsname = existing_datasets[0]
    dataset_dir = DATA_DIR.joinpath("generated/"+dsname)
    return dataset_dir, dsname

