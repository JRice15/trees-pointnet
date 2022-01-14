import argparse
import os
import sys
import time
import json
import glob
import traceback
from pprint import pprint
from pathlib import PurePath

# import geopandas as gpd
# import pandas as pd
# import laspy
import numpy as np
# import pyproj
# import shapely
import rasterio


# add parent directory
dn = os.path.dirname
sys.path.append(dn(dn(os.path.abspath(__file__))))

from core import DATA_DIR


# lazspark
with open("local_lazspark_path.txt", "r") as f:
    lazspark_path = f.read().strip()
print("lazspark path:", lazspark_path)
sys.path.append(lazspark_path)

from lazspark import stages



def process_region(regionname, spec, outname):
    outdir = DATA_DIR.joinpath("generated", outname, regionname, "lidar")

    globpath = DATA_DIR.joinpath("NAIP_patches/" + regionname.lower() + "_*.tif")
    all_bounds = []
    patch_ids = []
    for filename in glob.glob(globpath.as_posix()):
        patch_ids.append(int(PurePath(filename).stem.split("_")[0]))
        with rasterio.open(filename) as raster:
            all_bounds.append([i for i in raster.bounds])

    def key_func(arr, all_bounds, patch_ids):
        # initialize keys to -1
        out = -1 * np.ones(len(arr))
        for i, (left,bott,right,top) in enumerate(all_bounds):
            cond = np.logical_and.reduce(
                (arr >= left, arr >= bott, arr <= right, arr <= top)
            )
            out[cond] = patch_ids[i]
        return out
    
    rdd, header = stages.Reader(spec["lidar"], 1_000_000)()
    rdd_map = stages.Split(key_func, f_args=(all_bounds, patch_ids))(rdd)
    
    for patch_id, rdd in rdd_map:
        points = stages.Collect()(rdd)
        outfile = outdir.joinpath("lidar_patch_"+str(patch_id)+".npy").as_posix()
        np.save(outfile, points)
    



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
    os.makedirs(OUTDIR, exist_ok=True)

    statuses = {}

    for region_name, region_spec in data_specs.items():
        outdir = OUTDIR.joinpath(region_name)
        if os.path.exists(outfile) and not ARGS.overwrite:
            print("\n" + region_name, "already generated")
            statuses[region_name] = "Already present"
        else:
            print("\nGenerating region:", region_name)
            try:
                status = process_region(region_name, region_spec)
                print(status)
                statuses[region_name] = status
            except Exception as e:
                traceback.print_exc()
                statuses[region_name] = "Fail: " + str(e)

    pprint(statuses)

if __name__ == "__main__":
    main()
