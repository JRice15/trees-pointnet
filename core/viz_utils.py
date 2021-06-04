import contextlib
import time
import os

import h5py
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
import seaborn

seaborn.set()


def gaussian(x, center, sigma=0.02):
    const = (2 * np.pi * sigma) ** -0.5
    exp = np.exp( -np.sum((x - center) ** 2, axis=-1) / (2 * sigma ** 2))
    return const * exp

def raster_plot(pts, filename, weights=None, title=None, clip=None, scale=False):
    x = np.linspace(0, 1)
    y = np.linspace(0, 1)
    x, y = np.meshgrid(x, y)
    gridpts = np.stack([x,y], axis=-1)
    gridvals = np.zeros_like(x)
    for i,p in enumerate(pts):
        vals = gaussian(gridpts, p)
        if weights is not None:
            vals *= max(weights[i], 0)
        gridvals += vals

    if scale:
        gridvals = np.sqrt(gridvals)
    if clip is not None:
        gridvals = np.clip(gridvals, None, clip)

    plt.pcolormesh(x,y,gridvals, shading="auto")
    plt.colorbar()
    if title is not None:
        plt.title(title)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

