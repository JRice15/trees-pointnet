import argparse
from pprint import pprint
import time

import h5py
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras import backend as K
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from tensorflow import keras

np.random.seed(9999)
tf.random.set_seed(9999)
tf.compat.v1.set_random_seed(9999)

print("TF version:", tf.__version__)
print("Keras version:", keras.__version__)

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
