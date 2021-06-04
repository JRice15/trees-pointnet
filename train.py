import contextlib
import datetime
import json
import os
import argparse
from pprint import pprint
import time
import itertools

import h5py
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model
from tensorflow.keras import backend as K
from tensorflow.keras import callbacks, layers
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

from core import DATA_DIR, MAIN_DIR, ARGS, data_loading
from core.losses import get_loss
from core.models import pointnet
from core.utils import MyModelCheckpoint, output_model

"""
parse args
"""

modes_w_aliases = {
    "pwtt": ["pointwise-treetop"],
    "mmd": ["max-mean-discrepancy"],
    "count": [],
}

all_modes = list(modes_w_aliases.keys()) + list(itertools.chain.from_iterable(modes_w_aliases.values()))

parser = argparse.ArgumentParser()
parser.add_argument("--name",required=True,help="name to save this model under or load")
parser.add_argument("--mode",required=True,help="training mode, which determines which output flow and loss target to use",
    choices=all_modes)
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
parser.add_argument("--dropout",type=float,default=0.3,help="dropout rate")

# loss parameters
parser.add_argument("--mmd-sigma",type=float,default=0.05,
        help="max-mean-discrepancy mode: sigma on kernel")
parser.add_argument("--mmd-kernel",default="gaussian",
        help="max-mean-discrepancy mode: type of kernel")
parser.add_argument("--dist-weight",type=float,default=0.9,
        help="pointnet-treetop mode: weight on distance vs count loss")
parser.add_argument("--ortho-weight",type=float,default=0.001,
        help="orthogonality regularization loss weight")


ARGS = parser.parse_args(namespace=ARGS)

# manual args handling
if ARGS.test:
    ARGS.epochs = 2
    ARGS.batchsize = 2
for name,aliases in modes_w_aliases.items():
    if ARGS.mode in aliases:
        ARGS.mode = name
        break

if ARGS.mode in ["pwtt"]:
    ARGS.output_type = "seg"
elif ARGS.mode in ["count"]:
    ARGS.output_type = "cls"
elif ARGS.mode in ["mmd"]:
    ARGS.output_type = "custom"
else:
    raise ValueError("unknown mode to outputtype initialization")

pprint(vars(ARGS))

# create model output dir
now = datetime.datetime.now()
modelname = now.strftime("%y%m%d-%H%M%S-") + ARGS.name
MODEL_DIR = os.path.join(MAIN_DIR, "models/"+modelname)
os.makedirs(MODEL_DIR, exist_ok=False)
MODEL_PATH = os.path.join(MODEL_DIR, "model_" + ARGS.name + ".tf")

with open(os.path.join(MODEL_DIR, "params.json"), "w") as f:
    json.dump(vars(ARGS), f, indent=2)

"""
load data
"""

train_gen, val_gen = data_loading.get_train_val_gens(val_split=0.1, val_as_gen=ARGS.ragged)
train_gen.summary()
val_gen.summary()
inpt_shape = train_gen.get_batch_shape()[0][1:]

"""
create model
"""

# map modes to number of output features
output_pts_map = {
    "mmd": train_gen.max_trees,
    "pwtt": None,
    "count": None,
}
output_features_map = {
    "pwtt": 1,
    "count": 1,
    "mmd": 3, # (x,y,confidence)
}

model = pointnet(
    inpt_shape=inpt_shape,
    output_pts=output_pts_map[ARGS.mode],
    output_features=output_features_map[ARGS.mode],
    reg_weight=ARGS.ortho_weight,
)
output_model(model, MODEL_DIR)

loss, metrics = get_loss(ARGS)

model.compile(
    loss=loss, 
    metrics=metrics,
    optimizer=Adam(ARGS.lr)
)

callback_list = [
    callbacks.History(),
    callbacks.ReduceLROnPlateau(factor=ARGS.reducelr_factor, patience=ARGS.reducelr_patience,
        min_lr=1e-6, verbose=1),
    callbacks.EarlyStopping(verbose=1, patience=ARGS.reducelr_patience*2),
    MyModelCheckpoint(MODEL_PATH, verbose=1, epoch_per_save=5, save_best_only=True)
]

"""
train model
"""

if not ARGS.ragged:
    try:
        H = model.fit(
            x=train_gen,
            validation_data=val_gen.load_all(),
            epochs=ARGS.epochs,
            callbacks=callback_list,
            batch_size=ARGS.batchsize,
        )
    except KeyboardInterrupt:
        H = callback_list[0]

    os.makedirs(os.path.join(MODEL_DIR, "training"))
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
        start_time = time.time()

        step_time = 0
        batch_time = 0
        step_end_time = time.time()
        # Iterate over the batches of the dataset.
        for step, (x_batch_train, y_batch_train) in enumerate(train_gen):
            step_start_time = time.time()
            batch_time += step_start_time - step_end_time

            loss_values = train_step(x_batch_train, y_batch_train)
            # loss_values = model.train_on_batch(x_batch_train, y_batch_train)

            step_end_time = time.time()
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

        print("  Time taken: %.2fs" % (time.time() - start_time))

train_gen.close_file()
val_gen.close_file()


"""
Testing phase
"""
import evaluate

# add qualified name with timestamp as name, so it is unambigous in case of multiple models with same name
ARGS.name = modelname
evaluate.main()
