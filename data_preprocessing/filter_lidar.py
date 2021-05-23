import argparse
import json
import re
import os
from pprint import pprint

import numpy as np
import pdal

parser = argparse.ArgumentParser()
parser.add_argument("--infile",required=True)
parser.add_argument("--outfile",required=True)
parser.add_argument("--subsample",type=int,default=1,help="optional subsampling factor")
ARGS = parser.parse_args()

pipeline = []

pipeline.append(ARGS.infile)
pipeline.append({
    "type": "filters.hag_nn"
})
pipeline.append({
    "type": "filters.range",
    "limits": "HeightAboveGround[0.1:]" # hag >=0.1
})
if ARGS.subsample > 1:
    pipeline.append({
        "type": "filters.decimation",
        "step": ARGS.subsample,
    })
pipeline.append({
    "type": "writers.las",
    "compression": True,
    "extra_dims": "HeightAboveGround=float32",
    "filename": ARGS.outfile
})



pipeline_json = {
    "pipeline": pipeline
}

pipeline = pdal.Pipeline(json.dumps(pipeline_json))
pipeline.execute()
