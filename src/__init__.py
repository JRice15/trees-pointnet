"""
__init__.py: loads CL ARGS and sets random seeds seeds
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


try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import Model
    from tensorflow.keras import backend as K
    from tensorflow.keras import layers
    from tensorflow.keras.optimizers import Adam

    np.random.seed(9999)
    tf.random.set_seed(9999)
    tf.compat.v1.set_random_seed(9999)
    print("GPUs:", tf.config.list_physical_devices('GPU'))
except ModuleNotFoundError:
    pass
    # we don't always need tensorflow, so it's ok if we don't have it here
