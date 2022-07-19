"""
__init__.py: initializes constants and sets random seeds
"""

import argparse
import datetime
import json
import os
import random
import shutil
import sys
import time
from pathlib import PurePath
from pprint import pprint

import numpy as np

""" define global ARGS object """

ARGS = argparse.Namespace()


""" set global constants """

LIDAR_CHANNELS = ('x', 'y', 'height', 'red', 'green', 'blue', 'nir', 'ndvi')

from os.path import dirname as dirn
# up three steps: src -> pointnet -> tree-pointnet
REPO_ROOT = PurePath(dirn(dirn(dirn(__file__))))
DATA_DIR = REPO_ROOT.joinpath("data")
os.makedirs(REPO_ROOT.joinpath("pointnet/models"), exist_ok=True)
# add root to path so 'shared' dir is visible
sys.path.append(REPO_ROOT.as_posix())
# also add pointnet root
sys.path.append(REPO_ROOT.joinpath("pointnet").as_posix())

MODEL_SAVE_FMT = ".h5"

# custom keras layers
CUSTOM_LAYERS = {}

# random seeds
np.random.seed(999)
random.seed(999)

has_tf = False
try:
    import tensorflow as tf
    has_tf = True
except ModuleNotFoundError:
    pass
    # we don't always need tensorflow, so it's ok if we don't have it here


if has_tf:
    tf.get_logger().setLevel('ERROR')

    from tensorflow import keras

    tf.random.set_seed(999)
    tf.compat.v1.set_random_seed(999)
    gpus = tf.config.list_physical_devices('GPU')
    print("GPUs:", len(gpus))
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except Exception as e:
            print("! Cannot set memory growth on device", gpu)
            print(e)


