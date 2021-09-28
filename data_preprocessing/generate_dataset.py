import argparse
import os
import sys
import time
import json
from pathlib import PurePath

import geopandas as gpd
import pandas as pd
import h5py
import laspy
import numpy as np
# import pyproj
import shapely
import rasterio
# import tables
import numba


def naip2ndvi(im):
    nir = im[...,3]
    red = im[...,0]
    ndvi = (nir - red) / (nir + red)
    return ndvi


# @numba.njit
def seperate_pts(gt_bounds, x, y, z=None):
    """
    seperate points into the grid squares defined by gt_bounds, and limiting z to the 
    interval (0,120) when given
    args:
        gt_bounds: array where each element is of the form [left,bottom,right,top] bounds
        x, y, (z): coordinates of points (where optional z is heightaboveground)
    returns:
        list, where each index is the points that correspond to the gt_patch of the same index
    """
    out = []
    if z is None:
        pts = np.stack((x, y), axis=-1)
        for left,bottom,right,top in gt_bounds:
            out.append(
                pts[
                    (x >= left) & (x < right) & (y <= top) & (y > bottom) 
                ]
            )
    else:
        pts = np.stack((x, y, z), axis=-1)
        for left,bottom,right,top in gt_bounds:
            out.append(
                pts[
                    (x >= left) & (x < right) & (y <= top) & (y > bottom) & (z > 0) & (z < 120)  
                ]
            )
    return out
        


def load_lidar(las_file, grid_bounds):
    chunk_size = 500_000
    count = 0
    out = None
    with laspy.open(las_file, "r") as reader:
        while True:
            pts = reader.read_points(chunk_size)
            if pts is None:
                break

            # Note to future: never use pts["X"], only pts["x"]. the capitalized version scales the numbers to 
            # remove the decimal bc that's how laspy stores the underlying data
            x = pts["x"]
            y = pts["y"]
            z = pts["HeightAboveGround"]

            xyz_sep = seperate_pts(grid_bounds, np.array(x), np.array(y), np.array(z))
            if out is None:
                out = xyz_sep
            else:
                for i,existing_pts in enumerate(out):
                    out[i] = np.concatenate((existing_pts, xyz_sep[i]), axis=0)

            count += len(pts)
            print(count, "points loaded")

            # TEMP
            if count > 0:
                # print(out)
                return out

    return out


def load_grid_bounds(grid_filename, subdivide):
    """ 
    Load a EPSG:26911 grid definition
    args:
        subdivide: number of times to divide the side of each grid square. for example,
            subdivide=2 results in 4 squares per original, while subdivide=3 results in 9
    """
    grid = gpd.read_file(grid_filename)
    assert str(grid.crs).lower() == "epsg:26911"

    # the grid is perfectly aligned with the CRS, so the polygon bounds give us the
    # edges of each grid square
    grid_bounds = grid["geometry"].bounds.to_numpy()

    if subdivide > 1:
        new_grid_bounds = []
        for left,bottom,right,top in grid_bounds:
            x_width = (right - left) / subdivide
            y_width = (top - bottom) / subdivide
            for x in np.linspace(left, right, subdivide+1)[:-1]:
                for y in np.linspace(bottom, top, subdivide+1)[:-1]:
                    new_grid_bounds.append([x, y, x+x_width, y+y_width])
        grid_bounds = np.array(new_grid_bounds)

    return grid_bounds


def load_gt_trees(filename, gt_crs=None):
    """
    load gt tree locations from a .gpkg or .csv file.
    args:
        filename
        gt_crs: required if filename is a .csv
    """
    if filename.endswith(".gpkg"):
        gt = gpd.read_file(filename)

    elif filename.endswith(".csv"):
        if gt_crs is None:
            raise ValueError("Provide a 'gt_crs' key for this region")
        gt_raw = pd.read_csv(filename)
        gt_raw.columns = [x.lower() for x in gt_raw.columns]
        points = gpd.points_from_xy(gt_raw["longitude"], gt_raw["latitude"])
        gt = gpd.GeoDataFrame(geometry=points, crs=gt_crs)

    else:
        raise NotImplementedError("Cannot handle file {}".format(filename))

    gt = gt.to_crs("EPSG:26911")

    x = gt["geometry"].x
    y = gt["geometry"].y

    gt = np.stack([x, y], axis=1)

    return gt

def load_naip(naipfile, grid_bounds):

    raster = rasterio.open(naipfile)
    assert str(raster.crs).lower() == "epsg:26911"
    im = raster.read()
    im = np.moveaxis(im, 0, -1) / 256 # channels last, float
    assert im.shape[-1] == 4
    print("naip shape", im.shape)
    print("naip bounds", raster.bounds)

    # list of NAIP patches corresponding to test grid squares
    out = []
    for left,bottom,right,top in grid_bounds:
        y_min, x_min = raster.index(left, top)
        y_max, x_max = raster.index(right, bottom)
        patch = im[y_min:y_max, x_min:x_max]
        out.append(patch)

    return out, raster




def benchmark(name, _cache={}):
    """timing function"""
    t = time.perf_counter()
    if "last" in _cache:
        print("  ", name, t - _cache["last"])
    _cache["last"] = t
        

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--specs",required=True,help="json file with dataset specs")
    parser.add_argument("--outname",required=True,help="name of output h5 file (to be placed within the `../data` directory)")
    parser.add_argument("--subdivide",type=int,default=2,help="number of times to divide each grid square (resulting grid squares is subidivde**2 times the original")
    parser.add_argument("--overwrite",action="store_true")
    ARGS = parser.parse_args()

    with open(ARGS.specs, "r") as f:
        data_specs = json.load(f)

    OUTDIR = PurePath("../data/generated/{}".format(ARGS.outname))

    try:
        os.makedirs(OUTDIR)
    except FileExistsError:
        if not ARGS.overwrite:
            print("This will overwrite previously created dataset info stored under {}".format(OUTDIR))
            print("Press enter to continue, otherwise ctrl-C or D or whatever stops programs on your system", end=" ")
            input()

    with open(OUTDIR.joinpath("data_sources.json"), "w") as f:
        json.dump(data_specs, f, indent=2)

    with h5py.File(OUTDIR.joinpath("dataset.h5").as_posix(), "w") as hf:
        for region_name, region_spec in data_specs.items():
            print("\nLoading region:", region_name)

            benchmark("start")

            # grid definition
            grid_bounds = load_grid_bounds(region_spec["grid"], subdivide=ARGS.subdivide)
            benchmark("grid")

            # ground truth trees
            gt_crs = region_spec.get("gt_crs", None) # default to None if key doesn't exist
            gt_trees = load_gt_trees(region_spec["gt"], gt_crs=gt_crs)
            benchmark("gt")

            # remove grid squares that have no gt trees
            sep_gt_trees = seperate_pts(grid_bounds, gt_trees[:,0], gt_trees[:,1])
            valid_patch = [len(i) >= 3 for i in sep_gt_trees]
            grid_bounds = np.array([v for i,v in enumerate(grid_bounds) if valid_patch[i]])
            sep_gt_trees = [v for i,v in enumerate(sep_gt_trees) if valid_patch[i]]
            benchmark("grid filtering")

            # lidar points
            sep_lidar = load_lidar(region_spec["lidar"], grid_bounds)
            for i,lidar_patch in enumerate(sep_lidar):
                if len(lidar_patch) < 100: # fewer than 100 points in a patch
                    raise ValueError(
                        "Lidar patch {} with only {} points. Grid square bounds {}".format(i, len(lidar_patch), grid_bounds[i])
                    )
            benchmark("lidar")

            # naip
            naip_patches, naip_raster = load_naip(region_spec["naip"], grid_bounds)
            benchmark("load naip")

            # add NDVI channel to lidar
            lidar_w_naip = []
            for pt_group in sep_lidar:
                xys = pt_group[:,:2]
                naip = naip_raster.sample(xys) # returns generator
                # convert to float
                naip = np.array([i for i in naip]) / 256
                ndvi = naip2ndvi(naip).reshape(-1, 1)
                pt_group = np.concatenate([pt_group, ndvi], axis=-1)
                lidar_w_naip.append(pt_group)

            benchmark("add NDVI to lidar")

            # write to hdf5
            hf.create_dataset("/grids/{}".format(region_name), data=grid_bounds)
            for i in range(len(sep_gt_trees)):
                hf.create_dataset("/gt/{}_{}".format(region_name, i), data=sep_gt_trees[i])
                hf.create_dataset("/lidar/{}_{}".format(region_name, i), data=lidar_w_naip[i])
                hf.create_dataset("/naip/{}_{}".format(region_name, i), data=naip_patches[i])


if __name__ == "__main__":
    main()
