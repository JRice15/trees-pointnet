import argparse
import contextlib
import datetime
import glob
import itertools
import json
import math
import os
import shutil
import time
from pathlib import PurePath
from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model
from tensorflow.keras import backend as K
from tensorflow.keras import callbacks, layers, optimizers

import evaluate
from src import ARGS, DATA_DIR, MODEL_SAVE_FMT, REPO_ROOT
from src.losses import get_loss
from src.models import pointnet
from src.patch_generator import get_datasets
from src.tf_utils import MyModelCheckpoint, load_saved_model, output_model
from src.utils import Bounds, get_all_regions, get_default_dsname
from src.viz_utils import plot_one_example

"""
parse args
"""

optimizer_options = {
    "adam": optimizers.Adam,
    "adadelta": optimizers.Adadelta,
    "nadam": optimizers.Nadam,
    "adamax": optimizers.Adamax,
}

valid_output_flows = ["seg", "dense", "count"]
valid_losses = ["mmd", "gridmse", "p2p"]

parser = argparse.ArgumentParser(add_help=False)
# main required args
requiredgrp = parser.add_argument_group("required")
requiredgrp.add_argument("--name",required=True,help="name to save this model under or load")
requiredgrp.add_argument("--output-flow",required=True,help="which output flow to use",
    choices=valid_output_flows)
requiredgrp.add_argument("--loss",required=True,help="loss function to use",
    choices=valid_losses)

# main optional
optionalgrp = parser.add_argument_group("optional")
optionalgrp.add_argument("--eval",action="store_true",help="whether to evaluate at the end")
optionalgrp.add_argument("-h", "--help", action="help", help="show this message and exit")

# dataset
datagrp = parser.add_argument_group("dataset and augmentation")
datagrp.add_argument("--ragged",action="store_true")
datagrp.add_argument("--subdivide",type=int,default=3,help="number of times to subdivide standard grids. The side of a grid square is divided N times, resulting in N-squared new grid squares")
datagrp.add_argument("--regions",default="ALL",nargs="+",help="list of region names, defaults to all available")
datagrp.add_argument("--dsname",help="name of generated dataset to use (required if multiple exist)")
datagrp.add_argument("--noise-sigma",type=float,default=None,help="add gaussian noise to input points")
datagrp.add_argument("--handle-small",choices=["drop","fill","repeat"],default="drop",
    help="how to handle patches with fewer than npoints: drop them, fill with (-1000), or double up valid points until it meets the threshold")

# training hyperparameters
hypergrp = parser.add_argument_group("training hyperparameters")
hypergrp.add_argument("--optimizer",choices=list(optimizer_options.keys()),default="adam")
hypergrp.add_argument("--epochs",type=int,default=500)
hypergrp.add_argument("--batchsize",type=int,default=16)
hypergrp.add_argument("--lr",type=float,default=1e-3,help="initial learning rate")
hypergrp.add_argument("--reducelr-factor",type=float,default=0.2,help="factor to multiply lr by for reducelronplateau")
hypergrp.add_argument("--reducelr-patience",type=int,default=3,help="number of epochs with no valloss improvement to reduce lr")

# model parameters
modelgrp = parser.add_argument_group("model parameters")
modelgrp.add_argument("--npoints",type=int,default=500,help="number of points to run per patch. In ragged or non-ragged, "
        "patches with fewer points will be skipped. Also in non-ragged, patches with more points with be subsampled to npoints")
modelgrp.add_argument("--out-npoints",type=int,default=256,help="(dense output flow): number of output points")
modelgrp.add_argument("--size-multiplier",type=float,default=1.0,help="number to multiply all default conv output filters by")
modelgrp.add_argument("--dropout",dest="dropout_rate",type=float,default=0.0,help="dropout rate")
modelgrp.add_argument("--use-tnet1",action="store_true",help="whether to use input transform TNet")
modelgrp.add_argument("--use-tnet2",action="store_true",help="whether to use feature transform TNet")
modelgrp.add_argument("--conf-act",default="relu",help="activation to use on output confidences")

modelgrp.add_argument("--pnet2",dest="use_pnet2",action="store_true",help="use pointnet++ (aka pointnet2)")
modelgrp.add_argument("--batchnorm",dest="use_batchnorm",action="store_true",help="pnet2: whether to use batchnormalization")

# loss parameters
lossgrp = parser.add_argument_group("loss parameters")
lossgrp.add_argument("--gaussian-sigma", "--sigma",type=float,default=3,
        help="mmd/gridmse: in meters, std dev of gaussian smoothing applied in the loss")
lossgrp.add_argument("--mmd-kernel",default="gaussian",
        help="mmd: type of kernel")
lossgrp.add_argument("--gridmse-agg",choices=["max","sum"],default="sum",
        help="gridmse: how to aggregate predicted points during rasterization")

lossgrp.add_argument("--p2p-conf-weight",type=float,default=1.0, #TODO p2p defaults
        help="p2p: weight for confidence relative to distance in matching")
lossgrp.add_argument("--p2p-unmatched-weight",type=float,default=0.5, #TODO
        help="p2p: weight for loss on unmatched predictions (relative to matched) inside the classification loss")
lossgrp.add_argument("--p2p-loc-weight",type=float,default=1.0, #TODO
        help="p2p: weight for location (regression) loss, relative to classification loss")

lossgrp.add_argument("--ortho-weight",type=float,default=0.001,
        help="tnet2: orthogonality regularization loss weight, when using TNet2")

# misc
miscgrp = parser.add_argument_group("misc")
miscgrp.add_argument("--test",action="store_true",help="run minimal batches and epochs to test functionality")
miscgrp.add_argument("--noplot",action="store_true",help="no batch plots")
miscgrp.add_argument("--nolosses",action="store_true",help="eval only: do not compute losses")
miscgrp.add_argument("--show-summary",action="store_true",help="show model summary")


ARGS = parser.parse_args(namespace=ARGS)

assert ARGS.subdivide >= 1

# manual args handling
if ARGS.test:
    ARGS.epochs = 6
    ARGS.batchsize = 3

pprint(vars(ARGS))

# create model output dir
now = datetime.datetime.now()
modelname = ARGS.name + now.strftime("-%y%m%d-%H%M%S")
MODEL_DIR = REPO_ROOT.joinpath("models/"+modelname)
os.makedirs(MODEL_DIR, exist_ok=False)

if ARGS.dsname is None:
    ARGS.dsname = get_default_dsname()

if ARGS.regions == "ALL":
    ARGS.regions = get_all_regions(ARGS.dsname)

# save arguments to params file
with open(MODEL_DIR.joinpath("params.json"), "w") as f:
    json.dump(vars(ARGS), f, indent=2)


"""
load data
"""

train_gen, val_gen = get_datasets(ARGS.dsname, ARGS.regions, ("train", "val"))
train_gen.summary()
val_gen.summary()
x_shape, y_shape = train_gen.get_batch_shape()
inpt_shape = x_shape[1:]


if not ARGS.noplot:
    print("Generating example batch plots")
    train_viz_dir = MODEL_DIR.joinpath("training/example_batch_viz/")
    train_gen.__getitem__(0) # generate and throw away one batch, to make sure we don't have errors that dont appear the first time around
    X,Y,ids = train_gen.__getitem__(1, return_ids=True, no_rotate=True)
    print("For example batch:")
    print("X max:", tf.reduce_max(X, axis=[0,1]))
    print("X min:", tf.reduce_min(X, axis=[0,1]))
    print("Y max:", tf.reduce_max(Y, axis=[0,1]))
    print("Y min:", tf.reduce_min(Y, axis=[0,1]))
    # plot 8 examples
    for i in range(min(len(X), 8)):
        naip = train_gen.get_naip(ids[i])
        x = X[i].numpy()
        y = Y[i].numpy()
        plot_one_example(
            train_viz_dir.joinpath("normalized"),
            ids[i],
            X=x, Y=y, naip=naip, 
            bounds=Bounds.zero_to_one())

        x = train_gen.denormalize_pts(x, ids[i])
        y[:,:2] = train_gen.denormalize_pts(y[:,:2], ids[i])
        plot_one_example(
            train_viz_dir.joinpath("original_scale"),
            ids[i],
            X=x, Y=y, naip=naip,
            bounds=train_gen.get_patch_bounds(ids[i]))
            
        if ARGS.test:
            break


"""
create model
"""
print("Building model")

# map loss modes to number of output features and points
output_channels_map = {
    "gridmse": 3,   # x,y,confidence
    "mmd": 3,       # x,y,confidence
    "p2p": 3        # x,y,confidence
}

model = pointnet(
    inpt_shape=inpt_shape,
    size_multiplier=ARGS.size_multiplier,
    output_channels=output_channels_map[ARGS.loss],
)
output_model(model, MODEL_DIR, show=ARGS.show_summary)

loss, metrics = get_loss()

optim = optimizer_options[ARGS.optimizer]
model.compile(
    loss=loss, 
    metrics=metrics,
    optimizer=optim(ARGS.lr)
)

callback_dict = {
    "history": callbacks.History(),
    "reducelr": callbacks.ReduceLROnPlateau(factor=ARGS.reducelr_factor, patience=ARGS.reducelr_patience,
        min_lr=1e-6, verbose=1),
    "earlystopping": callbacks.EarlyStopping(verbose=1, patience=ARGS.reducelr_patience+2),
    "modelcheckpoint": MyModelCheckpoint(MODEL_DIR, verbose=1, 
        epoch_per_save=1, save_best_only=True)
}

"""
train model
"""

try:
    H = model.fit(
        x=train_gen,
        validation_data=val_gen.load_all(),
        validation_batch_size=ARGS.batchsize,
        epochs=ARGS.epochs,
        callbacks=list(callback_dict.values()),
        # batch_size=ARGS.batchsize,
    )
except KeyboardInterrupt:
    # allow manual stopping by user
    print("training ended manually...")
    H = callback_dict["history"]
except Exception as e:
    # otherwise log then throw the error
    import traceback

    # create signal file
    with open(MODEL_DIR.joinpath("training_failed.txt"), "w") as f:
        traceback.print_exc(file=f)
    raise e

os.makedirs(MODEL_DIR.joinpath("training"), exist_ok=True)

# save training data
with open(MODEL_DIR.joinpath("training/stats.json"), "w") as f:
    val_results = {
        "best_val_loss": callback_dict["modelcheckpoint"].best_val_loss()
    }
    json.dump(val_results, f, indent=2)
# save metric plots
for k in H.history.keys():
    if not k.startswith("val_"):
        plt.plot(H.history[k])
        plt.plot(H.history["val_"+k])
        plt.legend(['train', 'val'])
        plt.title(k)
        plt.xlabel('epoch')
        plt.savefig(os.path.join(MODEL_DIR, "training", k+".png"))
        plt.close()


del train_gen
del val_gen


if ARGS.eval:
    """
    Testing phase
    """

    # add qualified name with timestamp as name, so it is unambigous in case of multiple models with same name
    ARGS.name = modelname
    ARGS.save_preds = False
    evaluate.main()
