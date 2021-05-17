"""
__init__.py: loads CL args and sets random seeds seeds
"""

import argparse
from pprint import pprint
import time
import os

import h5py
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras import backend as K
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from tensorflow import keras

print("TF version:", tf.__version__)
print("Keras version:", keras.__version__)

""" handle args """

parser = argparse.ArgumentParser()
parser.add_argument("--mode",required=True)
parser.add_argument("--ragged",action="store_true")
parser.add_argument("--epochs",type=int,default=100)
parser.add_argument("--batchsize",type=int,default=16)
parser.add_argument("--dist-weight",type=float,default=0.5,help="pointnet-treetop mode: weight on distance vs sum loss")
args = parser.parse_args()
pprint(vars(args))

if not args.ragged:
    print("Defaulting to non-ragged mode")


if args.mode in ["pointwise-treetop"]:
    args.output_type = "seg"
elif args.mode in ["count"]:
    args.output_type = "cls"
else:
    raise ValueError("unknown mode to outputtype initialization")

""" set global constants """

MAIN_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(MAIN_DIR, "data")
OUTPUT_DIR = os.path.join(MAIN_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

np.random.seed(9999)
tf.random.set_seed(9999)
tf.compat.v1.set_random_seed(9999)
