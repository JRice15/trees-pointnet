"""
use pdal env
"""

import sys
from copy import copy
import argparse
import re
import glob
import time
import os
import subprocess
import json
import rasterio
from rasterio.windows import Window
import tempfile
from shapely.geometry import box
from pathlib import PurePath

dirn = os.path.dirname
sys.path.append(dirn(dirn(dirn(os.path.abspath(__file__)))))
from common.utils import MyTimer

def run_pdal(pipeline, nostream=False):
    with tempfile.NamedTemporaryFile('w', delete=False) as f:
        json.dump({"pipeline": pipeline}, f)
    print(f.name)
    cmd = ["pdal", "pipeline"]
    if nostream:
        cmd.append("--nostream")
    cmd.append(f.name)
    subprocess.run(cmd)


parser = argparse.ArgumentParser()
parser.add_argument('--las',nargs="+",required=True)
parser.add_argument("--naip-dir", required=True)
parser.add_argument("--out-format",required=True)
parser.add_argument("--nostream",action="store_true")
ARGS = parser.parse_args()

# get bounds of NAIP
all_bounds = {}
for naip_path in glob.glob(os.path.join(ARGS.naip_dir, "*.tif")):
    patchnum = int(PurePath(naip_path).stem.split("_")[-1])
    with rasterio.open(naip_path) as raster:
        left, bottom, right, top = raster.bounds
        all_bounds[patchnum] = ([left, right], [bottom, top])

# make sure every index from 0 to N exists
all_bounds = [str(all_bounds[i]) for i in range(len(all_bounds))]

# laz pipeline
laz_pipeline = list(ARGS.las)
laz_pipeline.append(
    {
        "type": "filters.crop",
        "bounds": all_bounds
    }
)
laz_pipeline.append(ARGS.out_format)

print('running laz pipeline...')
timer = MyTimer()
run_pdal(laz_pipeline, nostream=ARGS.nostream)

timer.measure("done")

# change numbering from 1 to 0 based
if "#" in ARGS.out_format:
    index = ARGS.out_format.index("#")
    prefix = ARGS.out_format[:index]
    suffix = ARGS.out_format[index+1:]
    globpath = re.sub(r"#", "*", ARGS.out_format)
    keyfunc = lambda x: (len(x), x)
    for path in sorted(glob.glob(globpath), key=keyfunc):
        num = int(path[len(prefix):-len(suffix)])
        target_num = num - 1
        assert target_num >= 0
        target_path = prefix + str(target_num) + suffix
        assert not os.path.exists(target_path), target_path
        os.rename(path, target_path)

