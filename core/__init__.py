"""
__init__.py: loads CL args and sets random seeds seeds
"""

import argparse
import datetime
import os
import shutil
import time
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

""" handle args """

parser = argparse.ArgumentParser()
parser.add_argument("--name",required=True,help="name to save this model under")
parser.add_argument("--mode",required=True)
parser.add_argument("--ragged",action="store_true")
parser.add_argument("--test",action="store_true",help="run minimal batches and epochs to test functionality")

# training hyperparameters
parser.add_argument("--epochs",type=int,default=250)
parser.add_argument("--batchsize",type=int,default=16)
parser.add_argument("--lr",type=float,default=0.003,help="initial learning rate")
parser.add_argument("--reducelr_factor",type=float,default=0.2,help="initial learning rate")
parser.add_argument("--reducelr_patience",type=float,default=50,help="initial learning rate")

# model parameters
parser.add_argument("--npoints",type=int,default=300,help="number of points to run per patch. In ragged or non-ragged, "
        "patches with fewer points will be skipped. Also in non-ragged, patches with more points with be truncated to npoints")
parser.add_argument("--dist-weight",type=float,default=0.5,
        help="pointnet-treetop mode: weight on distance vs sum loss")
parser.add_argument("--ortho-weight",type=float,default=0.001,
        help="orthogonality regularization loss weight")


args = parser.parse_args()
pprint(vars(args))

if args.mode in ["pointwise-treetop"]:
    args.output_type = "seg"
elif args.mode in ["count"]:
    args.output_type = "cls"
else:
    raise ValueError("unknown mode to outputtype initialization")

if args.test:
    args.epochs = 2
    args.batchsize = 2

""" set global constants """

MAIN_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(MAIN_DIR, "data")
OUTPUT_DIR = os.path.join(MAIN_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(MAIN_DIR, "models"), exist_ok=True)

now = datetime.datetime.now()
modelname = now.strftime("%y%m%d-%H%M%S-") + args.name
MODEL_DIR = os.path.join(MAIN_DIR, "models/"+modelname)
os.makedirs(MODEL_DIR, exist_ok=False)
MODEL_PATH = os.path.join(MODEL_DIR, "model_" + args.name + ".tf")

np.random.seed(9999)
tf.random.set_seed(9999)
tf.compat.v1.set_random_seed(9999)
