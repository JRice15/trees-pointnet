import contextlib
import glob
import os
import json
import time
from pathlib import PurePath

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import rasterio

from src import ARGS, DATA_DIR, REPO_ROOT, MODEL_SAVE_FMT


class Bounds:
    """
    unambigous bounds. Two formats available:
    xy: (min_x, max_x, min_y, max_y)
    minmax: (min_x, min_y, max_x, max_y)
    """

    def __init__(self, *, min_x, max_x, min_y, max_y):
        self.min_x = min_x
        self.max_x = max_x
        self.min_y = min_y
        self.max_y = max_y
    
    def minmax_fmt(self):
        return [self.min_x, self.min_y, self.max_x, self.max_y]

    def xy_fmt(self):
        return [self.min_x, self.max_x, self.min_y, self.max_y]
    
    @classmethod
    def from_xy(cls, bounds):
        """
        create Bounds object from an iterable `bounds` that is in xy format
        """
        keys = ("min_x", "max_x", "min_y", "max_y")
        kwargs = dict(zip(keys, bounds))
        return cls(**kwargs)

    @classmethod
    def from_minmax(cls, bounds):
        """
        create Bounds object from an iterable `bounds` that is in minmax format
        """
        keys = ("min_x", "min_y", "max_x", "max_y")
        kwargs = dict(zip(keys, bounds))
        return cls(**kwargs)


def gaussian(x, center, sigma=0.02):
    const = (2 * np.pi * sigma) ** -0.5
    exp = np.exp( -np.sum((x - center) ** 2, axis=-1) / (2 * sigma ** 2))
    return const * exp

def gridify_pts(bounds, pts, weights, abs_sigma=None, rel_sigma=None, mode="sum", 
        resolution=64):
    """
    rasterize weighted points to a grid
    args:
        bounds: Bounds object
        pts: (x,y) locations within the bounds
        weights: corresponding weights, (negative weights will be set to zero)
        {rel|abs}_sigma: specify relative (fraction of side length) or absolute distance sigma of gaussian smoothing kernel
        mode: how to aggregate values at each grid location. "max"|"sum"|"second-highest"
    returns:
        gridvals: N x M grid of weighted values
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
    if mode == "second-highest":
        gridvals = np.zeros(x.shape + (2,), dtype=x.dtype)
    else:
        gridvals = np.zeros_like(x)
    for i,p in enumerate(pts):
        new_vals = gaussian(gridpts, p, sigma=gaussian_sigma)
        if weights is not None:
            new_vals *= max(weights[i], 0)
        if mode == "sum":
            gridvals += new_vals
        elif mode == "max":
            gridvals = np.maximum(gridvals, new_vals)
        elif mode == "second-highest":
            # stack along 'channels' to make 3 channels
            gridvals = np.concatenate((gridvals, new_vals[..., None]), axis=-1)
            # get largest two from each 3-item channel
            gridvals = np.sort(gridvals, axis=-1)[..., -2:]
        else:
            raise ValueError("Unknown gridify_pts mode")
    if mode == "second-highest":
        return gridvals[...,-2], gridpts
    return gridvals, gridpts


def raster_plot(pts, filename, *, rel_sigma=None, abs_sigma=None, weights=None, title=None, clip=None, 
        sqrt_scale=False, mode="sum", mark=None, zero_one_bounds=False, weight_label=None):
    """
    create raster plot of points, with optional weights
    args:
        guassian_sigma: stddev of kernel to use when raster smoothing, where 1 is the width of the plot
        clip: None, or max value for output
        mode: max, or sum. method of generating y values in raster
        {rel|abs}_sigma: specify relative (fraction of side length) or absolute distance sigma of gaussian smoothing kernel
        mark: points to mark with a green x, of shape (n,2) where n is number of points
    """
    if zero_one_bounds:
        bounds = Bounds.from_xy([0, 1, 0, 1])
    else:
        bounds = Bounds.from_xy([
            pts[:,0].min(), pts[:,0].max(), 
            pts[:,1].min(), pts[:,1].max()
        ])

    gridvals, gridpts = gridify_pts(bounds, pts, weights, rel_sigma=rel_sigma, 
                            abs_sigma=abs_sigma, mode=mode)

    if sqrt_scale:
        gridvals = np.sqrt(gridvals)
    if clip is not None:
        gridvals = np.clip(gridvals, None, clip)

    x = gridpts[...,0]
    y = gridpts[...,1]
    plt.pcolormesh(x,y,gridvals, shading="auto")
    plt.colorbar(label=weight_label)

    if mark is not None:
        if isinstance(mark, dict):
            # colors which are well visible overlayed over blues/greens
            colors = ["red", "orange"]
            markers = ["x", "o", "^", "s"]
            for i,m in enumerate(mark.values()):
                plt.scatter(m[:,0], m[:,1], c=colors[i], marker=markers[i])
            plt.legend(mark.keys())
        else:
            plt.scatter(mark[:,0], mark[:,1], c="white", marker="x")

    if title is not None:
        plt.title(title)
    plt.tight_layout()
    plt.savefig(filename)
    plt.clf()
    plt.close()


def glob_modeldir(modelname):
    allmodels_dir = REPO_ROOT.joinpath("models/")

    # first try exact match
    matching_models = glob.glob(os.path.join(allmodels_dir.as_posix(), modelname+"-??????-??????", "model"+MODEL_SAVE_FMT))
    if len(matching_models) == 0:
        print("No exact model name matches")
        # then try autofill match
        matching_models = glob.glob(os.path.join(allmodels_dir.as_posix(), modelname+"*", "model"+MODEL_SAVE_FMT))
    
    if len(matching_models) > 1:
        print("Multiple models match 'name' argument:")
        print(" ", matching_models)
        print("Defaulting to the most recent:")
        # all the names have date/time string, so sorting gives order by time
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

def get_default_dsname():
    """
    get the dataset directory from name, or automatically select the one existing dataset if only one exists
    returns:
        dataset_dir, dsname
    """
    existing_datasets = [i for i in os.listdir(DATA_DIR.joinpath("lidar")) 
                        if os.path.isdir(DATA_DIR.joinpath("lidar", i))]
    if len(existing_datasets) > 1:
        raise ValueError("Multiple datasets exist in `data/lidar`. Specify which with the --dsname argument")
    elif len(existing_datasets) == 0:
        raise ValueError("No dataset exists in `data/lidar`")
    dsname = existing_datasets[0]
    return dsname



def rotate_pts(p, degrees=0):
    """
    in-place rotate points `p` counterclockwise by a multiple of 90 degrees, 
    around the point (0.5, 0.5)
    """
    if degrees == 0:
        return p
    origin = np.zeros_like(p)
    origin[...,:2] = 0.5
    p -= origin
    assert degrees % 90 == 0
    if degrees == 180:
        p[...,:2] = -p[...,:2]
    else:
        p[...,:2] = p[..., 1::-1]
        if degrees == 90:
            p[...,1] = -p[...,1]
        else:
            p[...,0] = -p[...,0]
    p += origin
    return p


def get_all_regions(dsname=None):
    if dsname is None:
        globpath = DATA_DIR.joinpath("gt_csvs", "*.csv").as_posix()
        files = glob.glob(globpath)
        regions = [PurePath(x).stem.split("_")[0] for x in files]
    else:
        globpath = DATA_DIR.joinpath("lidar", dsname, "regions", "*").as_posix()
        if globpath[-1] != "/":
            globpath += "/"
        regiondirs = glob.glob(globpath)
        regions = [PurePath(x).stem for x in regiondirs]
    if len(regions) < 1:
        raise ValueError("No regions found for lidar dataset {}".format(dsname))
    return regions

def get_naipfile_path(region, patch_num):
    """
    returns: str
    """
    filename = "{}_training_NAIP_NAD83_UTM11_{}.tif".format(region, patch_num)
    return DATA_DIR.joinpath("NAIP_patches", region, filename).as_posix()


def get_avg_patch_size():
    """
    get the average side length of the NAIP patches, in meters
    """
    stats_filename = DATA_DIR.joinpath("NAIP_patches", "computed_stats.json").as_posix()
    if not os.path.exists(stats_filename):
        # compute and save stats
        tifs = DATA_DIR.joinpath("NAIP_patches/*/*.tif").as_posix()
        sizes = []
        for tiffile in glob.glob(tifs):
            with rasterio.open(tiffile) as im:
                left,bott,right,top = [i for i in im.bounds]
                sizes += [right - left, top - bott]
        stats = {
            "avg_side_len_meters": np.mean(sizes)
        }
        with open(stats_filename, "w") as f:
            json.dump(stats, f, indent=2)
    # load and return stats
    with open(stats_filename, "r") as f:
        stats = json.load(f)
    return stats["avg_side_len_meters"]


def scaled_0_1(distance):
    """
    return the distance in meters scaled to approximate the
    the 0-1 normalized scale used during training
    """
    patch_len_meters = get_avg_patch_size() / ARGS.subdivide
    return distance / patch_len_meters

