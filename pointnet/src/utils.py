"""
general/misc utilities
"""

import contextlib
import glob
import os
import json
import time
from pathlib import PurePath

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
# import numba

from src import ARGS, DATA_DIR, REPO_ROOT, MODEL_SAVE_FMT

from common.data_handling import get_avg_patch_size


class LabeledBox:
    """
    a box storing an object, with an optional set of key-value pairs describing it
    """

    def __init__(self, obj, **params):
        self._obj = obj
        self._labels = labels

    def get(self):
        return self._obj

    def __getattr__(self, attr):
        return self._labels[attr]

    def set_labels(self, **labels):
        for k,v in labels.items():
            if k in self.labels:
                raise ValueError(f"{k} already in params")
            self.labels[k] = v
        return self

    def get_labels(self):
        return self._labels

    def copy(self):
        return LabeledObject(self.obj, **self.labels)



def load_params_into_ARGS(model_dir, ARGS, skip_params=(), false_params=()):
    """
    load original params into ARGS object
    args:
        skip_params: list of param keys to skip
        false_params: list of params to default to false, unless they are already set in ARGS
    """
    params_file = model_dir.joinpath("params.json")
    with open(params_file, "r") as f:
        params = json.load(f)
    for p in skip_params:
        params.pop(p)
    for p in false_params:
        params[p] = False
    for key,value in params.items():
        if not hasattr(ARGS, key):
            setattr(ARGS, key, value)


def glob_modeldir(modelname):
    """
    given a raw name (like "test-model"), find the most recent matching model
    path (eg, "models/test-model-220207-123423/")
    returns:
        pathlib.PurePath
    """
    allmodels_dir = REPO_ROOT.joinpath("pointnet/models/")

    # try match with fully qualified (timestamped) name
    exact_path = os.path.join(allmodels_dir.as_posix(), modelname)
    if os.path.exists(exact_path):
        return PurePath(exact_path)

    # first try exact name match
    matching_models = glob.glob(os.path.join(allmodels_dir.as_posix(), modelname+"-??????-??????"))
    if len(matching_models) == 0:
        print("No exact model name matches")
        # then try autofill match
        matching_models = glob.glob(os.path.join(allmodels_dir.as_posix(), modelname+"*"))
    
    if len(matching_models) > 1:
        print("Multiple models match 'name' argument.")
        print("Defaulting to the most recent:")
        # all the names have date/time string, so sorting gives order by time
        matching_models.sort()
        model_dir = PurePath(matching_models[-1])
    elif len(matching_models) == 0:
        raise FileNotFoundError("No matching models!")
    else:
        model_dir = PurePath(matching_models[0])

    return model_dir


def rotate_pts(p, degrees=0):
    """
    in-place rotate points `p` counterclockwise by a multiple of 90 degrees, 
    around the point (0.5, 0.5)
    """
    if degrees == 0:
        return p
    origin = np.zeros_like(p)
    origin[...,:2] = 0.5
    p -= origin
    assert degrees % 90 == 0
    if degrees == 180:
        p[...,:2] = -p[...,:2]
    else:
        p[...,:2] = p[..., 1::-1]
        if degrees == 90:
            p[...,1] = -p[...,1]
        else:
            p[...,0] = -p[...,0]
    p += origin
    return p


def scale_meters_to_0_1(meters, subdivide):
    """
    return the distance in meters scaled to approximate the
    the 0-1 normalized scale used during training
    """
    patch_len_meters = get_avg_patch_size() / subdivide
    return meters / patch_len_meters


def group_by_composite_key(d, first_n, agg_f=None):
    """
    given dict d, with keys that are tuples, group by selecting the first n elements
    of each key. Example with first_n=2:
    {
        ('a', 'b', 'c'): 1
        ('a', 'b', 'd'): 2
    } 
    -> 
    {
        ('a', 'b'): {
            ('c',): 1,
            ('d',): 2
        }
    }
    args:
        d: dict
        first_n: int, number of elements in key to groupby
        agg_f: None, or function that aggregates each subdict
    returns:
        dict of dicts, if agg_f is None
        dict of Any, if agg_f if function dict->Any
    """
    result = {}
    for key, val in d.items():
        start_key = key[:first_n]
        end_key = key[first_n:]
        if start_key not in result:
            result[start_key] = {}     
        result[start_key][end_key] = val
    if agg_f is not None:
        for key in list(result.keys()):
            result[key] = agg_f(result[key])
    return result



