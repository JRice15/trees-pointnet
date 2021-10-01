import argparse
import os
import sys
import time
import json
import traceback
from pprint import pprint
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
# import numba


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
            selected = pts[
                    (x >= left) & (x < right) & (y <= top) & (y > bottom) & (z > 0) & (z < 120)  
                ]
            out.append(selected)
    return out
        


def load_lidar(las_file, grid_bounds):
    chunk_size = 5_000_000
    count = 0
    out = None
    with laspy.open(las_file, "r") as reader:
        for pts in reader.chunk_iterator(chunk_size):
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
            print("   {:.3f}% complete: {} of {} lidar points".format(count/reader.header.point_count*100, count, reader.header.point_count))

            # # TEMP
            # if count > 0:
            #     # print(out)
            #     return out

    return out


def load_grid_bounds(grid_filename, subdivide):
    """ 
    Load a grid definition
    args:
        subdivide: number of times to divide the side of each grid square. for example,
            subdivide=2 results in 4 squares per original, while subdivide=3 results in 9
    """
    grid = gpd.read_file(grid_filename)

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

    return grid_bounds, str(grid.crs).lower()


def load_gt_trees(filename, target_crs, given_crs=None):
    """
    load gt tree locations from a .gpkg or .csv file.
    args:
        filename
        target_crs
        given_crs: required if filename is a .csv
    """
    if filename.endswith(".gpkg"):
        gt = gpd.read_file(filename)

    elif filename.endswith(".csv"):
        if given_crs is None:
            raise ValueError("Provide a 'gt_crs' key for this region")
        gt_raw = pd.read_csv(filename)
        gt_raw.columns = [x.lower() for x in gt_raw.columns]
        geom = gpd.points_from_xy(gt_raw["longitude"], gt_raw["latitude"])
        gt = gpd.GeoDataFrame(geometry=geom, crs=given_crs)
    else:
        raise NotImplementedError("Cannot handle file {}".format(filename))

    gt = gt.to_crs(target_crs)

    x = gt["geometry"].x
    y = gt["geometry"].y

    gt = np.stack([x, y], axis=1)

    return gt

def load_naip(naipfile, grid_bounds):
    """
    args:
        naipfile: filename
        grid bounds: np array of grid bounds
    returns:
        list of image patches corresponding to each grid square
        rasterio raster object
        raster crs
    """
    raster = rasterio.open(naipfile)
    im = raster.read()
    im = np.moveaxis(im, 0, -1) / 256 # channels last, float
    if im.shape[-1] != 4:
        raise ValueError("Wrong number of channels in NAIP. Expected 4, got {}".format(im.shape[-1]))
    print("naip shape", im.shape)
    print("naip bounds", raster.bounds)

    # list of NAIP patches corresponding to test grid squares
    out = []
    for left,bottom,right,top in grid_bounds:
        y_min, x_min = raster.index(left, top)
        y_max, x_max = raster.index(right, bottom)
        patch = im[y_min:y_max, x_min:x_max]
        out.append(patch)

    return out, raster, str(raster.crs).lower()




def benchmark(name, _cache={}):
    """timing function"""
    t = time.perf_counter()
    if "last" in _cache:
        print("  ", name, t - _cache["last"])
    _cache["last"] = t


def generate_region_h5(outfile, metafile, region_spec, subdivide=1):
    """
    load and format the data in an hdf5 outfile based on the region_specs
    returns:
        result status (str)
    """
    benchmark("start")

    # grid definition
    grid_bounds, grid_crs = load_grid_bounds(region_spec["grid"], subdivide=subdivide)
    benchmark("grid")

    # ground truth trees
    gt_crs = region_spec.get("gt_crs", None) # default to None if key doesn't exist
    gt_trees = load_gt_trees(region_spec["gt"], target_crs=grid_crs, given_crs=gt_crs)
    benchmark("gt")

    # remove grid squares that have no gt trees
    sep_gt_trees = seperate_pts(grid_bounds, gt_trees[:,0], gt_trees[:,1])
    valid_patch = [len(i) >= 3 for i in sep_gt_trees]
    grid_bounds = np.array([v for i,v in enumerate(grid_bounds) if valid_patch[i]])
    sep_gt_trees = [v for i,v in enumerate(sep_gt_trees) if valid_patch[i]]
    benchmark("grid filtering")

    # naip
    naip_patches, naip_raster, naip_crs = load_naip(region_spec["naip"], grid_bounds)
    benchmark("load naip")

    if grid_crs != naip_crs:
        raise ValueError("grid and NAIP crs do not match: {} {}".format(grid_crs, naip_crs))

    # lidar points
    sep_lidar = load_lidar(region_spec["lidar"], grid_bounds)
    for i,lidar_patch in enumerate(sep_lidar):
        if len(lidar_patch) < 100: # fewer than 100 points in a patch
            print("! Lidar patch {} with only {} points. Grid square bounds {}".format(i, len(lidar_patch), grid_bounds[i]))
            return "Fail: too few lidar points in a grid square"
    benchmark("lidar")

    # add NDVI channel to lidar
    lidar_w_naip = []
    n_groups = len(sep_lidar)
    for n,pt_group in enumerate(sep_lidar):
        print("Adding NDVI", n, "of", n_groups)
        xys = pt_group[:,:2]
        naip = naip_raster.sample(xys) # returns generator
        # convert to float
        naip = np.array([i for i in naip]) / 256
        ndvi = naip2ndvi(naip).reshape(-1, 1)
        pt_group = np.concatenate([pt_group, ndvi], axis=-1)
        lidar_w_naip.append(pt_group)

    naip_raster.close()
    benchmark("add NDVI to lidar")

    meta = {
        "crs": grid_crs,
        "subdivide": subdivide,
        "spec": region_spec
    }
    with open(metafile.as_posix(), "w") as f:
        json.dump(meta, f, indent=2)

    with h5py.File(outfile.as_posix(), "w") as hf:
        # write to hdf5
        hf.create_dataset("/grid", data=grid_bounds)
        for i in range(len(sep_gt_trees)):
            hf.create_dataset("/gt/patch{}".format(i), data=sep_gt_trees[i])
            hf.create_dataset("/lidar/patch{}".format(i), data=lidar_w_naip[i])
            hf.create_dataset("/naip/patch{}".format(i), data=naip_patches[i])

    benchmark("write to file")
    return "Success"



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--specs",required=True,help="json file with dataset specs")
    parser.add_argument("--outname",required=True,help="name of output h5 file (to be placed within the `../data` directory)")
    parser.add_argument("--subdivide",type=int,default=2,help="number of times to divide each grid square (resulting grid squares is subidivde**2 times the original")
    parser.add_argument("--overwrite",action="store_true")
    ARGS = parser.parse_args()

    with open(ARGS.specs, "r") as f:
        data_specs = json.load(f)

    # make sure no spaces in region name keys
    if any([" " in i for i in data_specs.keys()]):
        raise ValueError("No spaces allowed in specs region name keys")

    OUTDIR = PurePath("../data/generated/{}".format(ARGS.outname))
    os.makedirs(OUTDIR, exist_ok=True)

    statuses = {}

    for region_name, region_spec in data_specs.items():
        outfile = OUTDIR.joinpath(region_name + ".h5")
        metafile = OUTDIR.joinpath(region_name + "_meta.json")
        if os.path.exists(outfile) and not ARGS.overwrite:
            print("\n" + region_name, "already generated")
            statuses[region_name] = "Already present"
        else:
            print("\nLoading region:", region_name)
            try:
                status = generate_region_h5(outfile, metafile, region_spec, subdivide=ARGS.subdivide)
                print(status)
                statuses[region_name] = status
            except Exception as e:
                traceback.print_exc()
                statuses[region_name] = "Fail: " + str(e)

    pprint(statuses)

if __name__ == "__main__":
    main()
