"""
general/misc utilities
"""

import contextlib
import glob
import os
import json
import time
from pathlib import PurePath

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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
        self.min_x = float(min_x)
        self.max_x = float(max_x)
        self.min_y = float(min_y)
        self.max_y = float(max_y)
    
    def __repr__(self):
        return f"Bounds(xmin={self.min_x}, xmax={self.max_x}, ymin={self.min_y}, ymax={self.max_y}"

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

    @classmethod
    def zero_to_one(cls):
        """
        create a Bounds object signifying bounds from 0 to 1 in the X and Y directions
        """
        return Bounds.from_minmax([0, 0, 1, 1])


class LabeledBox:
    """
    a box storing an object, with an optional set of key-value pairs describing it
    """

    def __init__(self, obj, **params):
        self._obj = obj
        self._labels = labels

    def get(self):
        return self._obj

    def __getattr__(self, attr):
        return self._labels[attr]

    def set_labels(self, **labels):
        for k,v in labels.items():
            if k in self.labels:
                raise ValueError(f"{k} already in params")
            self.labels[k] = v
        return self

    def get_labels(self):
        return self._labels

    def copy(self):
        return LabeledObject(self.obj, **self.labels)



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
    """
    given a raw name (like "test-model"), find the most recent matching model
    path (eg, "models/test-model-220207-123423/")
    returns:
        pathlib.PurePath
    """
    allmodels_dir = REPO_ROOT.joinpath("models/")

    # try match with fully qualified (timestamped) name
    exact_path = os.path.join(allmodels_dir.as_posix(), modelname)
    if os.path.exists(exact_path):
        return PurePath(exact_path)

    # first try exact name match
    matching_models = glob.glob(os.path.join(allmodels_dir.as_posix(), modelname+"-??????-??????"))
    if len(matching_models) == 0:
        print("No exact model name matches")
        # then try autofill match
        matching_models = glob.glob(os.path.join(allmodels_dir.as_posix(), modelname+"*"))
    
    if len(matching_models) > 1:
        print("Multiple models match 'name' argument.")
        print("Defaulting to the most recent:")
        # all the names have date/time string, so sorting gives order by time
        matching_models.sort()
        model_dir = PurePath(matching_models[-1])
    elif len(matching_models) == 0:
        raise FileNotFoundError("No matching models!")
    else:
        model_dir = PurePath(matching_models[0])

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

def load_naip(region, patch_num, bounds=None):
    """
    load a NAIP tile, or a subsection of it
    args:
        region: str
        patch_num: int
        bounds: Bounds object, to select a subwindow of the NAIP
    """
    naipfile = get_naipfile_path(region, patch_num)
    with rasterio.open(naipfile) as raster:
        if bounds is None:
            im = raster.read()
        else:
            # https://gis.stackexchange.com/questions/336874/get-a-window-from-a-raster-in-rasterio-using-coordinates-instead-of-row-column-o
            im = raster.read(window=rasterio.windows.from_bounds(*bounds.minmax_fmt(), raster.transform))
    # channels last format
    im = np.moveaxis(im, 0, -1) / 255.0
    # sometimes it selects a row of pixels outside of the image, which results in spurious very large negative numbers
    im = np.clip(im, 0, 1)
    return im


def get_naip_bounds(region, patch_num):
    """
    get the bounds of a NAIP tile specified by a region and patch_number
    returns:
        Bounds
    """
    naipfile = get_naipfile_path(region, patch_num)
    with rasterio.open(naipfile) as raster:
        return Bounds.from_minmax(raster.bounds)


def load_gt_trees(region):
    """
    load all ground-truth annotations from a region
    args:
        region: str
    """
    x_column_options = ("long_nad83", "long_utm11", "point_x", "x")
    y_column_options = ("lat_nad83", "lat_utm11", "point_y", "y")

    GT_DIR = DATA_DIR.joinpath("ground_truth_csvs")
    files = glob.glob(GT_DIR.joinpath(region + "*.csv").as_posix())
    assert len(files) == 1, f"found {len(files)} instead of 1"
    table = pd.read_csv(files[0])

    # lowercase the columns
    table.columns = [x.lower() for x in table.columns]
    x_col = None
    y_col = None
    for col in x_column_options:
        if col in table:
            x_col = table[col].to_numpy()
            break
    for col in y_column_options:
        if col in table:
            y_col = table[col].to_numpy()
            break
    if x_col is None or y_col is None:
        raise ValueError("Could not find correct gt columns for {}".format(region))
    gt_pts = np.stack([x_col, y_col], axis=-1)
    return gt_pts


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


def scale_meters_to_0_1(meters, subdivide):
    """
    return the distance in meters scaled to approximate the
    the 0-1 normalized scale used during training
    """
    patch_len_meters = get_avg_patch_size() / subdivide
    return meters / patch_len_meters


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



