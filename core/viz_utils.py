import contextlib
import time
import os

import h5py
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
import seaborn



def gaussian(x, center, sigma=0.02):
    const = (2 * np.pi * sigma) ** -0.5
    exp = np.exp( -np.sum((x - center) ** 2, axis=-1) / (2 * sigma ** 2))
    return const * exp



def raster_plot(pts, filename, gaussian_sigma, weights=None, title=None, clip=None, 
        sqrt_scale=False, mode="sum", mark=None):
    """
    create raster plot of points, with optional weights
    args:
        guassian_sigma: stddev of kernel to use when raster smoothing, where 1 is the width of the plot
        clip: None, or max value for output
        mode: max, or sum. method of generating y values in raster
        mark: points to mark with a green x, of shape (n,2) where n is number of points
    """
    x = np.linspace(pts[:,0].min(), pts[:,0].max())
    y = np.linspace(pts[:,1].min(), pts[:,1].max())
    size_factor = x.max() - x.min()
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
            raise ValueError("Unknown raster_plot mode")

    if sqrt_scale:
        gridvals = np.sqrt(gridvals)
    if clip is not None:
        gridvals = np.clip(gridvals, None, clip)

    plt.pcolormesh(x,y,gridvals, shading="auto")
    plt.colorbar()

    if mark is not None:
        plt.scatter(mark[:,0], mark[:,1], c="white", marker="x")

    if title is not None:
        plt.title(title)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

