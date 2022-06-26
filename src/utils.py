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
# import numba

from src import ARGS, DATA_DIR, REPO_ROOT, MODEL_SAVE_FMT

class MyTimer:

    def __init__(self, msg_indent=2, decimals=4):
        self.msg_indent = msg_indent
        self.decimals = decimals
        self.start()
    
    def start(self):
        self._start_time = time.perf_counter()
    
    def measure(self, name=None):
        elapsed = time.perf_counter() - self._start_time
        prefix = " " * self.msg_indent
        if name is not None:
            prefix += str(name) + ": "
        if elapsed > 60:
            elapsed /= 60
            unit = "min"
        else:
            unit = "sec"
        elapsed = round(elapsed, self.decimals)
        print(f"{prefix}{elapsed} {unit}")
        self.start()


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
    
    def filter_pts(self, pts, min_z=None, max_z=None):
        """
        return only the points that fall within this bounds. optional z filters
        """
        x = pts[:,0]
        y = pts[:,1]
        # boolean condition
        mask = (x >= self.min_x) & (x < self.max_x) & \
               (y >= self.min_y) & (y < self.max_y)
        if min_z is not None:
            z = pts[:,2]
            mask = mask & (z >= min_z)
        if max_z is not None:
            z = pts[:,2]
            mask = mask & (z < max_z)
        return pts[mask]

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

# @numba.njit
def gaussian(x, center, sigma=0.02):
    """
    args:
        x: locations to evaluate gassian curve at
        center: peak of gaussian
    """
    const = (2 * np.pi * sigma) ** -0.5
    exp = np.exp( -np.sum((x - center) ** 2, axis=-1) / (2 * sigma ** 2))
    return const * exp

# @numba.njit
# def _gridify_loop(gridvals, gridpts, pts, weights, gaussian_sigma, mode):
#     for i,p in enumerate(pts):
#         if weights[i] > 0:
#             new_vals = gaussian(gridpts, p, sigma=gaussian_sigma)
#             new_vals *= weights[i]

#             if mode == "sum":
#                 gridvals += new_vals
#             elif mode == "max":
#                 gridvals = np.maximum(gridvals, new_vals)
#             elif mode == "median" or mode == "second-highest":
#                 # stack along 'channels'
#                 gridvals = np.concatenate((gridvals, new_vals[..., None]), axis=-1)
#             else:
#                 raise ValueError("Unknown gridify_pts mode")

#     return gridvals


def gridify_pts(bounds, pts, weights, abs_sigma=None, rel_sigma=None, mode="sum", 
        resolution=None):
    """
    rasterize weighted points to a grid
    args:
        bounds: Bounds object
        pts: (x,y) locations within the bounds
        weights: corresponding weights, (negative weights will be set to zero)
        {rel|abs}_sigma: specify relative (fraction of side length) or absolute distance sigma of gaussian smoothing kernel
        mode: how to aggregate values at each grid location. options: max, sum, second-highest, median
    returns:
        gridvals: N x M grid of weighted values
        gridpts: N x M x 2 grid, representing the x,y coordinates of each pixel
    """
    if resolution is None:
        resolution = 64

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

    # initialize grid for aggregation methods
    if mode == "second-highest" or mode == "median":
        gridvals_list = []
    else:
        gridvals = np.zeros_like(x)

    # filter out small weights: 
    # 1/100th of max weight or 1e-3, whichever is smaller
    min_weight = min(0.01 * max(weights), 1e-3)
    mask = (weights > min_weight)
    pts = pts[mask]
    weights = weights[mask]

    for point,weight in zip(pts, weights):
        new_vals = weight * gaussian(gridpts, point, sigma=gaussian_sigma)

        if mode == "sum":
            gridvals += new_vals
        elif mode == "max":
            gridvals = np.maximum(gridvals, new_vals)
        elif mode == "median" or mode == "second-highest":
            # collect all values
            gridvals_list.append(new_vals)
        else:
            raise ValueError("Unknown gridify_pts mode")
    
    if mode == "second-highest":
        gridvals = np.stack(gridvals_list, axis=0)
        # get second largest along channels dim
        gridvals = np.sort(gridvals, axis=0)[-2]
    elif mode == "median":
        gridvals = np.stack(gridvals_list, axis=0)
        # get median along channels dim 
        gridvals = np.median(gridvals, axis=-0)

    return gridvals, gridpts



def plot_raster(gridvals, gridcoords, filename, *, colorbar_label=None, 
        mark=None, title=None, grid_resolution=None, ticks=False, colorbar=True):
    """
    create plot of a raster
    args:
        title: plot title
        colorbar_label: label on colorbar
        mark: dict mapping names to points to mark with an x, array shape (n,2)
    """
    x = gridcoords[...,0]
    y = gridcoords[...,1]
    plt.pcolormesh(x,y,gridvals, shading="auto")
    if colorbar:
        plt.colorbar(label=colorbar_label)

    if mark is not None:
        if isinstance(mark, dict):
            # colors which are well visible overlayed over blues/greens
            colors = ["red", "orange"]
            markers = ["x", "o", "^", "s"]
            for i,m in enumerate(mark.values()):
                plt.scatter(m[:,0], m[:,1], c=colors[i], marker=markers[i])
            plt.legend(mark.keys(), loc="upper left")
        else:
            plt.scatter(mark[:,0], mark[:,1], c="white", marker="x")

    if title is not None:
        plt.title(title)
    if not ticks:
        plt.xticks([])
        plt.yticks([])
    plt.tight_layout()
    plt.savefig(filename)
    plt.clf()
    plt.close()


def rasterize_and_plot(pts, filename, *, rel_sigma=None, abs_sigma=None, weights=None, 
        title=None, clip=None, sqrt_scale=False, mode="sum", mark=None, 
        zero_one_bounds=False, weight_label=None, grid_resolution=None,
        colorbar=True, ticks=False):
    """
    create raster plot of points, with optional weights
    args:
        guassian_sigma: stddev of kernel to use when raster smoothing, where 1 is the width of the plot
        clip: None, or max value for output
        mode: max, or sum. method of generating y values in raster
        {rel|abs}_sigma: specify relative (fraction of side length) or absolute distance sigma of gaussian smoothing kernel
        mark: dict mapping names to points to mark with an x, array shape (n,2)
    """
    if zero_one_bounds:
        bounds = Bounds.from_xy([0, 1, 0, 1])
    else:
        bounds = Bounds.from_xy([
            pts[:,0].min(), pts[:,0].max(), 
            pts[:,1].min(), pts[:,1].max()
        ])

    gridvals, gridcoords = gridify_pts(bounds, pts, weights, rel_sigma=rel_sigma, 
                            abs_sigma=abs_sigma, mode=mode, resolution=grid_resolution)

    if sqrt_scale:
        gridvals = np.sqrt(gridvals)
    if clip is not None:
        gridvals = np.clip(gridvals, None, clip)

    plot_raster(gridvals, gridcoords, filename, 
        colorbar_label=weight_label,
        mark=mark,
        title=None,
        colorbar=colorbar,
        ticks=ticks)




def plot_one_example(outdir, patch_id, *, Y, X=None, pred=None, pred_peaks=None, 
        pred_overlap_gridded=None, naip=None, zero_one_bounds=False,
        grid_resolution=None):
    """
    generate raster plots for one example input and output from a dataset
    args:
        x: input from patch generator
        y: targets from patch generator
        patch_id
        outdir: pathlib.PurePath
        pred: raw predictions from network
        pred_overlap_gridded: pre-gridded predictions, generated by the overlap method. Expects same output from gridify_preds()
        pred_peaks: thresholded peaks from the predictions blurred to grid; ie the true final predictions
        naip: naip image
        zero_one_bounds: whether points are in 0-1 scale (instead of epsg26911)
    """
    if not os.path.exists(outdir):
        os.makedirs(outdir, exist_ok=True)

    patchname = "_".join([str(i) for i in patch_id])

    if Y.shape[-1] > 2:
        ylocs = Y[Y[...,2] > 0.5][...,:2]
    else:
        ylocs = Y

    gt_ntrees = len(ylocs)

    if zero_one_bounds:
        sigma = scaled_0_1(ARGS.gaussian_sigma)
    else:
        sigma = ARGS.gaussian_sigma

    markings = {"gt trees": ylocs}
    if pred_peaks is not None:
        markings["predicted trees"] = pred_peaks[...,:2]

    if X is not None:
        x_locs = X[...,:2]

        # lidar height (median mode, to avoid noise)
        x_heights = X[...,2]
        rasterize_and_plot(
            x_locs, 
            weights=x_heights, 
            weight_label="height",
            abs_sigma=sigma,
            mode="median", 
            filename=outdir.joinpath("{}_lidar_height".format(patchname)), 
            mark=markings, 
            zero_one_bounds=zero_one_bounds,
            grid_resolution=grid_resolution)
        
        # lidar ndvi
        x_ndvi = X[...,-1]
        rasterize_and_plot(
            x_locs, 
            weights=x_ndvi, 
            weight_label="ndvi", 
            mode="median",
            abs_sigma=sigma, 
            filename=outdir.joinpath("{}_lidar_ndvi".format(patchname)),
            mark=markings, 
            zero_one_bounds=zero_one_bounds,
            grid_resolution=grid_resolution)

    # raw predictions
    if pred is not None:
        # prediction confidence raster
        pred_locs = pred[...,:2]
        pred_weights = pred[...,2]
        rasterize_and_plot(
            pred_locs, 
            weights=pred_weights, 
            weight_label="prediction confidence",
            abs_sigma=sigma, 
            filename=outdir.joinpath("{}_pred_raw".format(patchname)),
            mode="sum", 
            mark=markings, 
            zero_one_bounds=zero_one_bounds,
            grid_resolution=grid_resolution)

    # overlap gridded predictions
    if pred_overlap_gridded is not None:
        plot_raster(
            pred_overlap_gridded["vals"], 
            pred_overlap_gridded["coords"], 
            colorbar_label="prediction confidence",
            mark=markings,
            filename=outdir.joinpath("{}_pred_overlapped".format(patchname))
        )

    if naip is not None:
        plt.imshow(naip[...,:3]) # only use RGB
        # plt.colorbar()
        plt.xticks([])
        plt.yticks([])
        plt.tight_layout()
        plt.savefig(outdir.joinpath(patchname+"_NAIP_RGB.png"))
        plt.clf()
        plt.close()


def load_params_into_ARGS(model_dir, ARGS, skip_params=(), false_params=()):
    """
    load original params into ARGS object
    args:
        skip_params: list of param keys to skip
        false_params: list of params to set to false
    """
    params_file = model_dir.joinpath("params.json")
    with open(params_file, "r") as f:
        params = json.load(f)
    for p in skip_params:
        params.pop(p)
    for p in false_params:
        params[p] = False
    for key,value in params.items():
        if not hasattr(ARGS, key):
            setattr(ARGS, key, value)


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
        raise FileNotFoundError("No matching models!")
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
    globname = "{}_*_{}.tif".format(region, patch_num)
    globpath = DATA_DIR.joinpath("NAIP_patches/", region, globname).as_posix()
    found = glob.glob(globpath)
    if len(found) != 1:
        raise ValueError("{} matching NAIP files found for '{} {}'. Expected exactly 1".format(len(found), region, patch_num))
    return found[0]


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


def group_by_composite_key(d, first_n, agg_f=None):
    """
    given dict d, with keys that are tuples, group by selecting the first n elements
    of each key. Example with first_n=2:
    {
        ('a', 'b', 'c'): 1
        ('a', 'b', 'd'): 2
    } 
    -> 
    {
        ('a', 'b'): {
            ('c',): 1,
            ('d',): 2
        }
    }
    args:
        d: dict
        first_n: int, number of elements in key to groupby
        agg_f: None, or function that aggregates each subdict
    returns:
        dict of dicts, if agg_f is None
        dict of Any, if agg_f if function dict->Any
    """
    result = {}
    for key, val in d.items():
        start_key = key[:first_n]
        end_key = key[first_n:]
        if start_key not in result:
            result[start_key] = {}     
        result[start_key][end_key] = val
    if agg_f is not None:
        for key in list(result.keys()):
            result[key] = agg_f(result[key])
    return result



