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

import h5py
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model
from tensorflow.keras import backend as K
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam

print("TF version:", tf.__version__)
print("Keras version:", keras.__version__)

""" define global ARGS object """

ARGS = argparse.Namespace()


""" set global constants """

MAIN_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(MAIN_DIR, "data")
os.makedirs(os.path.join(MAIN_DIR, "models"), exist_ok=True)

np.random.seed(9999)
tf.random.set_seed(9999)
tf.compat.v1.set_random_seed(9999)
