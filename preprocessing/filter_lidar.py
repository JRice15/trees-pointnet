"""
- Create a HeightAboveGround (HAG) dimension from the Z component
# - filter to HAG>=0.0
- optionally subsample (take every nth point)
"""
import argparse
import json
import re
import os
from pprint import pprint

import numpy as np
import pdal

parser = argparse.ArgumentParser()
parser.add_argument("--infile",required=True)
parser.add_argument("--outfile",default=None,help="optional outfile name, defaults to same directory as infile with generated suffix")
parser.add_argument("--reproject",default=None,help="optional CRS string to reproject to")
parser.add_argument("--subsample",type=int,default=1,help="optional subsampling factor")
parser.add_argument("--scale",type=float,default=0.01,help="output scale (precision)")
ARGS = parser.parse_args()

if ARGS.outfile is None:
    if ARGS.infile[-4:] in (".laz", ".las"):
        outfile = ARGS.infile[:-4] + "_HAG"
        if ARGS.reproject:
            outfile += "_" + ARGS.reproject
        if ARGS.subsample > 1:
            outfile += "_subsample{}x".format(ARGS.subsample)
        outfile += ".laz"
        ARGS.outfile = outfile
    else:
        raise ValueError("unrecognized file extension, not .las or .laz")

pipeline = []

print("Reading   ", ARGS.infile)
print("Writing to", ARGS.outfile)

pipeline.append(ARGS.infile)
if ARGS.subsample > 1:
    pipeline.append({
        "type": "filters.decimation",
        "step": ARGS.subsample,
    })
if ARGS.reproject is not None:
    pipeline.append({
        "type": "filters.reprojection",
        "out_srs": ARGS.reproject,
    })
pipeline.append({
    "type": "filters.hag_nn"
})
# pipeline.append({
#     "type": "filters.range",
#     "limits": "HeightAboveGround[0.0:]" # hag >=0.0
# })
pipeline.append({
    "type": "writers.las",
    "scale_x": ARGS.scale,
    "scale_y": ARGS.scale,
    "scale_z": 0.01,
    "offset_x": "auto",
    "offset_y": "auto",
    "offset_z": "auto",
    "compression": True,
    "extra_dims": "HeightAboveGround=float32",
    "filename": ARGS.outfile
})



pipeline_json = {
    "pipeline": pipeline
}

pipeline = pdal.Pipeline(json.dumps(pipeline_json))
pipeline.execute()
