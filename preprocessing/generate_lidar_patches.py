import argparse
import os
import sys
import time
import json
import glob
import traceback
from pprint import pprint
from pathlib import PurePath

from tqdm import tqdm
# import geopandas as gpd
# import pandas as pd
import laspy
import numpy as np
# import pyproj
# import shapely
import rasterio

# add parent directory
dn = os.path.dirname
sys.path.append(dn(dn(os.path.abspath(__file__))))

from core import DATA_DIR

def naip2ndvi(im):
    nir = im[...,3]
    red = im[...,0]
    ndvi = (nir - red) / (nir + red)
    ndvi = np.nan_to_num(ndvi, nan=0.0)
    return ndvi


def load_lidar(las_file, patch_bounds, out_dict):
    chunk_size = 5_000_000

    print("  reading", las_file)
    with laspy.open(las_file, "r") as reader:
        for pts in tqdm(reader.chunk_iterator(chunk_size), total=(reader.header.point_count//chunk_size)):
            # Note to future: never use pts["X"], only pts["x"]. the capitalized version scales the numbers to 
            # remove the decimal bc that's how laspy stores the underlying data
            x = pts["x"]
            y = pts["y"]
            z = pts["HeightAboveGround"]
            pts = np.stack((x,y,z), axis=-1)

            for patch_id, (left,bott,right,top) in patch_bounds.items():
                cond = np.logical_and.reduce(
                    (x >= left, y >= bott, x <= right, y <= top)
                )
                selected = pts[cond]
                if len(selected):
                    if patch_id in out_dict:
                        out_dict[patch_id] = np.concatenate((out_dict[patch_id], selected))
                    else:
                        out_dict[patch_id] = selected

    return out_dict


def add_ndvi_to_pts(seperated_lidar, raster_map):
    out = {}
    for patch_id, pts in tqdm(seperated_lidar.items()):
        raster = raster_map[patch_id]
        xys = pts[:,:2]
        naip = raster.sample(xys) # returns generator
        # convert to float
        naip = np.array([val for val in naip]) / 256
        assert len(naip) == len(pts)
        ndvi = naip2ndvi(naip).reshape(-1, 1)
        pts = np.concatenate([pts, ndvi], axis=-1)
        out[patch_id] = pts
    return out


def process_region(regionname, spec, outdir):
    os.makedirs(outdir, exist_ok=True)

    globpath = DATA_DIR.joinpath("NAIP_patches/" + regionname.lower() + "_*.tif")
    bounds = {}
    rasters = {}
    for filename in glob.glob(globpath.as_posix()):
        patch_id = int(PurePath(filename).stem.split("_")[-1])
        raster = rasterio.open(filename)
        rasters[patch_id] = raster
        bounds[patch_id] = [i for i in raster.bounds]
    
    if not isinstance(spec["lidar"], list):
        spec["lidar"] = [spec["lidar"]]
    seperated_pts = {}
    for lidarfile in spec["lidar"]:
        seperated_pts = load_lidar(lidarfile, bounds, seperated_pts)

    seperated_pts = add_ndvi_to_pts(seperated_pts, rasters)

    for patch_id, pts in seperated_pts.items():
        outfile = outdir.joinpath("lidar_patch_"+str(patch_id)+".npy").as_posix()
        np.save(outfile, pts)
    
    for raster in rasters.values():
        raster.close()

    expected_keys = set(rasters.keys())
    outputted_keys = set(seperated_pts.keys())
    diff = expected_keys.difference(outputted_keys)
    if len(diff):
        missing = ", ".join([str(x) for x in diff])
        return "Missing outputs: " + missing

    return "Success"



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--specs",required=True,help="json file with dataset specs")
    parser.add_argument("--outname",required=True,help="name of output h5 file (to be placed within the `../data` directory)")
    parser.add_argument("--overwrite",action="store_true")
    ARGS = parser.parse_args()

    with open(ARGS.specs, "r") as f:
        data_specs = json.load(f)

    statuses = {}

    for region_name, region_spec in data_specs.items():
        print("\nGenerating region:", region_name)
        OUTDIR = DATA_DIR.joinpath("lidar", outname, regionname)
        # check for existing outputs
        existing_outputs = glob.glob(OUTDIR.joinpath("*.npy").as_posix())
        if len(existing_outputs) and not overwrite:
            statuses[region_name] = "Already exists"
            continue
        # generate data
        try:
            status = process_region(region_name, region_spec, OUTDIR)
            print(status)
            statuses[region_name] = status
            # save specs on success for future reference
            with open(OUTDIR.joinpath("specs.json"), "w") as f:
                json.dump(data_specs[region_name], f, indent=2)
        except Exception as e:
            traceback.print_exc()
            statuses[region_name] = "Fail: " + str(e)

    pprint(statuses)

if __name__ == "__main__":
    main()


