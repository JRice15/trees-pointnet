import json
import os
import sys
import time
import glob
from pathlib import PurePath

import numpy as np
import pandas as pd
import rasterio

from common import REPO_ROOT, DATA_DIR



"""
Useful data classes
"""

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


"""
train val test splitting
"""

def get_data_splits(sets=("train", "val", "test"), regions=None):
    """
    returns patch ids for the train, val, and test datasets
    it selects the same patches every time given the same split, by selecting 
    every Nth patch from a deterministically shuffled list from each region
    args:
        regions: list(str), or None which defaults ot all regions
    """
    with open(REPO_ROOT.joinpath("traintest_split.py"), "r") as f:
        splits = json.load(f)

    # select desired splits
    splits = [splits[name] for name in sets]

    if regions is not None:
        # filter by regions
        splits = [
            [p_id for p_id in set_ids if p_id[0] in regions] 
            for set_ids in splits
        ]

    return splits

"""
Data loading & utils
"""

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



def get_all_patch_ids(regions=None):
    """
    get all patch ids as specified by the NAIP tiles
    args:
        regions: list(str), or None (which defaults to all regions)
    """
    if regions is None:
        regions = get_all_regions()
    p_ids = []
    for region in regions:
        naipfiles_path = DATA_DIR.joinpath("NAIP_patches", region, "*.tif").as_posix()
        naipfiles = sorted(glob.glob(naipfiles_path))
        for filename in naipfiles:
            patch_num = int(PurePath(filename).stem.split("_")[-1])
            p_ids.append( (region, patch_num) )
    return p_ids


def get_all_regions(dsname=None):
    """
    get all regions as specified by the NAIP tiles, or a particular lidar dataset
    """
    if dsname is None:
        globpath = DATA_DIR.joinpath("ground_truth_csvs", "*.csv").as_posix()
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


def load_gt_trees_region(region):
    """
    load all raw ground-truth annotations from a region
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


def load_gt_trees(patch_ids):
    """
    load all ground truth trees, in proper patches
    args:
        regions: list(str)
    returns:
        dict mapping patch_id to tree locations array, shape (N,2)
    """
    regions = list(set([p_id[0] for p_id in patch_ids]))
    raw_trees = {r: load_gt_trees_region(r) for r in regions}
    patched_trees = {}
    for region, patch_num in patch_ids:
        bounds = get_naip_bounds(region, patch_num)
        patched_trees[(region, patch_num)] = bounds.filter_pts(raw_trees[region])
    return patched_trees


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
