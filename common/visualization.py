"""
utilities related to visualization & plotting
"""

import glob
import os
import json
import time
from pathlib import PurePath

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors
import numpy as np

from common import DATA_DIR

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=256):
    """
    https://stackoverflow.com/questions/18926031/how-to-extract-a-subset-of-a-colormap-as-a-new-colormap-in-matplotlib
    """
    new_cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap


CMAP = truncate_colormap(plt.get_cmap("viridis"), maxval=0.95)

# plot marker styles
ALL_MARKER_STYLE = {
    "edgecolors": 'black',
    "linewidths": 0.5,
}
GT_POS_MARKER_STYLE = {
    "marker": "o",
    "s": 65,
    "color": "gold",
}
GT_NEG_MARKER_STYLE = {
    "marker": "o",
    "s": 60,
    "color": "red",
}
TP_MARKER_STYLE = {
    "marker": "P",
    "s": 60,
    "color": "gold",
}
FP_MARKER_STYLE = {
    "marker": "X",
    "s": 60,
    "color": "red",
}

MARKER_STYLE_MAP = {
    # first key is gt/pred
    "gt": {
        # second key is whether these are correctly or incorrectly labelled by the model
        # None means we don't have correctness info available
        None: GT_POS_MARKER_STYLE,
        True: GT_POS_MARKER_STYLE,
        False: GT_NEG_MARKER_STYLE,
    },
    "pred": {
        None: TP_MARKER_STYLE,
        True: TP_MARKER_STYLE,
        False: FP_MARKER_STYLE,
    }
}

MARKER_LABEL_MAP = {
    "gt": {
        None: "ground-truth trees",
        True: "ground-truth tp",
        False: "ground-truth fn",
    },
    "pred": {
        None: "predicted trees",
        True: "predicted tp",
        False: "predicted fp"
    }
}


def plot_markers(marker_dict, legend=True):
    """
    plot xy tree locations (predicted or actual) on a raster
    args:
        pts: shape (N,3)
        kind: str, either "gt" or "pred"
        correct: True, False, or None. Whether these are correct or incorrect for the model
    """
    if marker_dict is not None:
        for (kind,correct), pts in marker_dict.items():
            if len(pts):
                style = MARKER_STYLE_MAP[kind][correct]
                label = MARKER_LABEL_MAP[kind][correct]
                plt.scatter(
                    pts[:,0], pts[:,1], 
                    label=label,
                    **style,
                    **ALL_MARKER_STYLE,
                )
        if legend and any(map(len, marker_dict.values())):
            plt.legend()



def make_marker_dict(gt=None, preds=None, pointmatch_inds=None):
    """
    make dictionary holding markers data
    keys are 2-tuple: (kind,correct)
    kind is one of "gt", "pred"
    correct is one of True, False, None
    """
    if pointmatch_inds is None:
        output = {}
        if gt is not None:
            output[("gt",None)] = gt
        if preds is not None:
            output[("pred",None)] = preds
    else:
        tp_gt = np.delete(gt, pointmatch_inds["fn"], axis=0)
        fn_gt = gt[pointmatch_inds["fn"].astype(int)]
        tp_pred = preds[pointmatch_inds["tp"].astype(int)]
        fp_pred = preds[pointmatch_inds["fp"].astype(int)]

        output = {
            ("gt",True): tp_gt,
            ("gt",False): fn_gt,
            ("pred",True): tp_pred,
            ("pred",False): fp_pred,
        }
    return output


def plot_NAIP(naip, bounds, filename, markers):
    """
    plot NAIP image with markers on gt/pred, optionally with pointmatching indexes
    to determine tp/fp/fn
    args:
        naip: (H,W,3+) image
        bounds: Bounds object
        filename: filename to save plot under (.png recommended)
        markers: markers dict
    """
    fig, ax = plt.subplots(figsize=(8,8))

    plt.imshow(
        naip[...,:3], # only use RGB
        extent=bounds.xy_fmt())
    plt.xticks([])
    plt.yticks([])

    plot_markers(markers)

    plt.tight_layout()
    plt.savefig(filename)
    plt.clf()
    plt.close()



def plot_raster(gridvals, gridcoords, filename, *, colorbar_label=None, 
        markers=None, title=None, grid_resolution=None, ticks=False, colorbar=True):
    """
    create plot of a raster (for creating the raster from raw pts, see rasterize_and_plot)
    args:
        title: plot title
        colorbar_label: label on colorbar
        markers: markers dict
    """
    shape = (10,8) if colorbar else (8,8)
    fig, ax = plt.subplots(figsize=shape)

    x = gridcoords[...,0]
    y = gridcoords[...,1]
    plt.pcolormesh(x,y,gridvals, shading="auto", cmap=CMAP)
    if colorbar:
        plt.colorbar(label=colorbar_label)

    plot_markers(markers)

    if title is not None:
        plt.title(title)
    if not ticks:
        plt.xticks([])
        plt.yticks([])
    plt.tight_layout()
    plt.savefig(filename)
    plt.clf()
    plt.close()

