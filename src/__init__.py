"""
__init__.py: initializes constants and sets random seeds
"""

import argparse
import datetime
import os
import shutil
import time
import json
from pprint import pprint
from pathlib import PurePath

import numpy as np


""" define global ARGS object """

ARGS = argparse.Namespace()


""" set global constants """

REPO_ROOT = PurePath(os.path.dirname(os.path.dirname(__file__)))
DATA_DIR = REPO_ROOT.joinpath("data")
os.makedirs(REPO_ROOT.joinpath("models"), exist_ok=True)

MODEL_SAVE_FMT = ".h5"

np.random.seed(9999)


has_tf = False
try:
    import tensorflow as tf
    has_tf = True
except ModuleNotFoundError:
    pass
    # we don't always need tensorflow, so it's ok if we don't have it here


if has_tf:
    from tensorflow import keras
    from tensorflow.keras import Model
    from tensorflow.keras import backend as K
    from tensorflow.keras import layers
    from tensorflow.keras.optimizers import Adam

    tf.random.set_seed(9999)
    tf.compat.v1.set_random_seed(9999)
    gpus = tf.config.list_physical_devices('GPU')
    print("GPUs:", gpus)
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except Exception as e:
            print("! Cannot set memory growth on device", gpu)
            print(e)

