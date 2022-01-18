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
                if patch_id in out_dict:
                    out_dict[patch_id] = np.concatenate((out_dict[patch_id], selected))
                else:
                    out_dict[patch_id] = selected

    return out_dict


def add_ndvi_to_pts(seperated_lidar, raster_map):
    out = {}
    for patch_id, pts in seperated_lidar.items():
        raster = rasters[patch_id]
        xys = pts[:,:2]
        naip = raster.sample(xys) # returns generator
        # convert to float
        naip = np.array([val for val in naip]) / 256
        assert len(naip) == len(pts)
        ndvi = naip2ndvi(naip).reshape(-1, 1)
        pts = np.concatenate([pts, ndvi], axis=-1)
        out[patch_id] = pts
    return out


def process_region(regionname, spec, outname):
    outdir = DATA_DIR.joinpath("generated", "lidar", outname, regionname)
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
    
    for raster in rasters.items():
        raster.close()

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

    OUTDIR = PurePath("../data/generated/lidar/{}".format(ARGS.outname))

    if os.path.exists(OUTDIR) and not ARGS.overwrite:
        raise FileExistsError("lidar patch dataset {} already exists".format(ARGS.outname))

    os.makedirs(OUTDIR, exist_ok=True)
    # save specs for future reference
    with open(OUTDIR.joinpath("specs.json"), "w") as f:
        json.dump(data_specs, f)

    statuses = {}

    for region_name, region_spec in data_specs.items():
        print("\nGenerating region:", region_name)
        try:
            status = process_region(region_name, region_spec, ARGS.outname)
            print(status)
            statuses[region_name] = status
        except Exception as e:
            traceback.print_exc()
            statuses[region_name] = "Fail: " + str(e)

    pprint(statuses)

if __name__ == "__main__":
    main()
