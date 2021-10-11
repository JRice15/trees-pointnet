import contextlib
import datetime
import json
import glob
import os
import argparse
from pprint import pprint
from pathlib import PurePath
import time
import shutil
import itertools

import h5py
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model
from tensorflow.keras import backend as K
from tensorflow.keras import callbacks, layers
from tensorflow.keras import optimizers
import matplotlib.pyplot as plt

from core import DATA_DIR, REPO_ROOT, ARGS, data_loading
from core.losses import get_loss
from core.models import pointnet
from core.tf_utils import MyModelCheckpoint, output_model

"""
parse args
"""

optimizer_options = {
    "adam": optimizers.Adam,
    "adadelta": optimizers.Adadelta,
    "nadam": optimizers.Nadam,
    "adamax": optimizers.Adamax,
}

valid_modes = ["pwtt", "mmd", "count", "pwmmd"]

parser = argparse.ArgumentParser(add_help=False)
# main required args
requiredgrp = parser.add_argument_group("required")
requiredgrp.add_argument("--name",required=True,help="name to save this model under or load")
requiredgrp.add_argument("--mode",required=True,help="training mode, which determines which output flow and loss target to use",
    choices=valid_modes)
requiredgrp.add_argument("--dsname",required=True,help="name of generated dataset")

# main optional
optionalgrp = parser.add_argument_group("optional")
optionalgrp.add_argument("--ragged",action="store_true")
optionalgrp.add_argument("--regions",default="ALL",nargs="+",help="list of region names, defaults to all available")
optionalgrp.add_argument("-h", "--help", action="help", help="show this message and exit")

# training hyperparameters
hypergrp = parser.add_argument_group("training hyperparameters")
hypergrp.add_argument("--optimizer",choices=list(optimizer_options.keys()),default="adam")
hypergrp.add_argument("--epochs",type=int,default=500)
hypergrp.add_argument("--batchsize",type=int,default=16)
hypergrp.add_argument("--lr",type=float,default=0.001,help="initial learning rate")
hypergrp.add_argument("--reducelr-factor",type=float,default=0.2,help="factor to multiply lr by for reducelronplateau")
hypergrp.add_argument("--reducelr-patience",type=int,default=20,help="number of epochs with no valloss improvement to reduce lr")

# model parameters
modelgrp = parser.add_argument_group("model parameters")
modelgrp.add_argument("--npoints",type=int,default=300,help="number of points to run per patch. In ragged or non-ragged, "
        "patches with fewer points will be skipped. Also in non-ragged, patches with more points with be truncated to npoints")
modelgrp.add_argument("--dropout",type=float,default=0.3,help="dropout rate")
modelgrp.add_argument("--ndvi",action="store_true",help="whether to use pointwise NDVi channel")

# loss parameters
lossgrp = parser.add_argument_group("loss parameters")
lossgrp.add_argument("--mmd-sigma",type=float,default=0.04,
        help="max-mean-discrepancy mode: sigma on kernel")
lossgrp.add_argument("--mmd-kernel",default="gaussian",
        help="max-mean-discrepancy mode: type of kernel")
lossgrp.add_argument("--dist-weight",type=float,default=0.9,
        help="pointnet-treetop mode: weight on distance vs count loss")
lossgrp.add_argument("--ortho-weight",type=float,default=0.001,
        help="orthogonality regularization loss weight")

# misc
miscgrp = parser.add_argument_group("misc")
miscgrp.add_argument("--test",action="store_true",help="run minimal batches and epochs to test functionality")


ARGS = parser.parse_args(namespace=ARGS)

# manual args handling
if ARGS.test:
    ARGS.epochs = 6
    ARGS.batchsize = 2

pprint(vars(ARGS))

# create model output dir
now = datetime.datetime.now()
modelname = ARGS.name + now.strftime("-%y%m%d-%H%M%S")
MODEL_DIR = REPO_ROOT.joinpath("models/"+modelname)
os.makedirs(MODEL_DIR, exist_ok=False)
MODEL_PATH = MODEL_DIR.joinpath("model_" + ARGS.name + ".tf")
DATASET_DIR = DATA_DIR.joinpath("generated/"+ARGS.dsname)
if ARGS.regions == "ALL":
    regions = glob.glob(os.path.join(DATASET_DIR.as_posix(), "*"))
    ARGS.regions = [PurePath(r).stem for r in regions]

with open(MODEL_DIR.joinpath("params.json"), "w") as f:
    json.dump(vars(ARGS), f, indent=2)


"""
load data
"""

train_gen, val_gen = data_loading.get_train_val_gens(DATASET_DIR, ARGS.regions, val_split=0.1, test_split=0.1)
train_gen.summary()
val_gen.summary()
inpt_shape = train_gen.get_batch_shape()[0][1:]

"""
create model
"""

# map modes to number of output features and points
output_channels_map = {
    "pwtt": 1,  # confidence, xys are appended
    "count": 1, # count
    "mmd": 3,   # x,y,confidence
    "pwmmd": 3, # x,y,confidence
}

model = pointnet(
    inpt_shape=inpt_shape,
    output_channels=output_channels_map[ARGS.mode],
    reg_weight=ARGS.ortho_weight,
)
output_model(model, MODEL_DIR)

loss, metrics = get_loss(ARGS)

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
    "earlystopping": callbacks.EarlyStopping(verbose=1, patience=int(ARGS.reducelr_patience*2.5)),
    "modelcheckpoint": MyModelCheckpoint(MODEL_PATH, verbose=1, 
        epoch_per_save=(5 if not ARGS.test else 1), save_best_only=True)
}

"""
train model
"""

if not ARGS.ragged:
    try:
        H = model.fit(
            x=train_gen,
            validation_data=val_gen.load_all(),
            epochs=ARGS.epochs,
            callbacks=list(callback_dict.values()),
            batch_size=ARGS.batchsize,
        )
    except KeyboardInterrupt:
        # allow manual stopping by user
        H = callback_dict["history"]
    except Exception as e:
        # otherwise log then throw the error
        import traceback
        # create signal file
        with open(MODEL_DIR.joinpath("training_failed.txt"), "w") as f:
            traceback.print_exc(file=f)
        raise e

    # evaluate on validation set
    VAL_DIR = MODEL_DIR.joinpath("results_val")
    val_gen.evaluate_model(model, VAL_DIR)

    # save training data
    os.makedirs(MODEL_DIR.joinpath("training"))
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

else:
    """
    adapted from https://keras.io/guides/writing_a_training_loop_from_scratch/
    """

    @tf.function
    def train_step(x, y):
        with tf.GradientTape() as tape:
            logits = model(x, training=True)
            loss_values = model.loss(y, logits)
            loss_values += model.losses
        grads = tape.gradient(loss_values, model.trainable_weights)
        model.optimizer.apply_gradients(zip(grads, model.trainable_weights))
        for m in model.metrics:
            m.update_state(y, logits)
        return loss_values

    @tf.function
    def test_step(x, y):
        logits = model(x, training=False)
        loss_values = model.loss(y, logits)
        loss_values += np.sum(model.losses)
        for m in model.metrics:
            m.update_state(y, logits)
        return loss_values

    def eval_metrics():
        metric_vals = [m.result() for m in model.metrics]
        return ", ".join(["{} {:.5f}".format(name, val) for name, val in zip(model.metrics_names, metric_vals)])

    LOG_FREQ = 20 # in batches

    for epoch in range(ARGS.epochs):
        print("Epoch {}".format(epoch))
        start_time = time.perf_counter()

        step_time = 0
        batch_time = 0
        step_end_time = start_time
        # Iterate over the batches of the dataset.
        for step, (x_batch_train, y_batch_train) in enumerate(train_gen):
            step_start_time = time.perf_counter()
            batch_time += step_start_time - step_end_time

            loss_values = train_step(x_batch_train, y_batch_train)
            # loss_values = model.train_on_batch(x_batch_train, y_batch_train)

            step_end_time = time.perf_counter()
            step_time += step_end_time - step_start_time
            # Log every n batches.
            if step % LOG_FREQ == 0:
                print("   {:>3d} -- avg time: step {:.3f}, batch {:.3f} -- loss {:.5f}, {}".format(
                    step, step_time/LOG_FREQ, batch_time/LOG_FREQ, np.mean(loss_values), eval_metrics()))
                step_time = 0
                batch_time = 0

        # Train metrics
        print("Train -- loss {:.5f}, {}".format(np.mean(loss_values), eval_metrics()))
        for m in model.metrics:
            m.reset_states()
            
        # Validation
        for x_batch_val, y_batch_val in val_gen:
            val_loss = test_step(x_batch_val, y_batch_val)
        print("Val   -- loss {:.5f}, {}".format(np.mean(val_loss), eval_metrics()))
        for m in model.metrics:
            m.reset_states()
        for c in callback_list:
            c.on_epoch_end(epoch, logs={"val_loss": np.mean(val_loss)})

        print("  Time taken: %.2fs" % (time.perf_counter() - start_time))

del train_gen
del val_gen


"""
Testing phase
"""
import evaluate

# add qualified name with timestamp as name, so it is unambigous in case of multiple models with same name
ARGS.name = modelname
evaluate.main()
