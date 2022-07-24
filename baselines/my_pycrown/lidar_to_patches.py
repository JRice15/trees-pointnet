"""
use pdal env
"""

from copy import copy
import argparse
import glob
import os
import subprocess
import json
import rasterio
from rasterio.windows import Window
import tempfile
from shapely.geometry import box
from pathlib import PurePath

def run_pdal(pipeline):
    with tempfile.NamedTemporaryFile('w', delete=False) as f:
        json.dump({"pipeline": pipeline}, f)
    print(f.name)
    subprocess.run(['pdal', 'pipeline', f.name])


parser = argparse.ArgumentParser()
parser.add_argument('--las',nargs="+",required=True)
parser.add_argument("--naip-dir", required=True)
parser.add_argument("--out-format",required=True)
ARGS = parser.parse_args()

assert "#" in ARGS.out_format

# get bounds of NAIP
all_bounds = {}
for naip_path in glob.glob(ARGS.naip_dir):
    patchnum = int(PurePath(naip_path).stem.split("_")[-1])
    with rasterio.open(naip_path) as raster:
        left, bott, right, top = raster.bounds
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
laz_pipeline.append(
    {
        "type": "writers.las",
        "filename": ARGS.out_format,
        "compression":"lazperf"
    })

print('running laz pipeline...')
run_pdal(laz_pipeline)

print("done")