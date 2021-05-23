import contextlib
import datetime
import json
import os
import time

import h5py
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model
from tensorflow.keras import backend as K
from tensorflow.keras import callbacks, layers
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

from core import DATA_DIR, OUTPUT_DIR, MAIN_DIR, args, data_loading
from core.losses import get_loss
from core.models import pointnet
from core.utils import MyModelCheckpoint, output_model

"""
handle args
"""

now = datetime.datetime.now()
modelname = now.strftime("%y%m%d-%H%M%S-") + args.name
MODEL_DIR = os.path.join(MAIN_DIR, "models/"+modelname)
os.makedirs(MODEL_DIR, exist_ok=False)
MODEL_PATH = os.path.join(MODEL_DIR, "model_" + args.name + ".tf")

if args.mode is None:
    print("'--mode' argument required")
    exit()
elif args.mode in ["pointwise-treetop"]:
    args.output_type = "seg"
elif args.mode in ["count"]:
    args.output_type = "cls"
else:
    raise ValueError("unknown mode to outputtype initialization")

with open(os.path.join(MODEL_DIR, "params.json"), "w") as f:
    json.dump(vars(args), f, indent=2)

"""
load data
"""

train_gen, val_gen = data_loading.get_train_val_gens(val_split=0.1, val_as_gen=args.ragged)
train_gen.summary()
val_gen.summary()
inpt_shape = train_gen.get_batch_shape()[1:]

"""
create model
"""

# map modes to number of output features
output_features_map = {
    "pointwise-treetop": 1,
    "count": 1,
}

model = pointnet(
    inpt_shape=inpt_shape,
    output_features=output_features_map[args.mode],
    reg_weight=args.ortho_weight,
)
output_model(model)

loss, metrics = get_loss(args)

model.compile(
    loss=loss, 
    metrics=metrics,
    optimizer=Adam(args.lr)
)

callback_list = [
    callbacks.History(),
    callbacks.ReduceLROnPlateau(factor=args.reducelr_factor, patience=args.reducelr_patience,
        min_lr=1e-6, verbose=1),
    callbacks.EarlyStopping(verbose=1, patience=args.reducelr_patience*2),
    MyModelCheckpoint(MODEL_PATH, verbose=1, epoch_per_save=5, save_best_only=True)
]

"""
train model
"""

if not args.ragged:
    try:
        H = model.fit(
            x=train_gen,
            validation_data=val_gen.load_all(),
            epochs=args.epochs,
            callbacks=callback_list,
            batch_size=args.batchsize,
        )
    except KeyboardInterrupt:
        H = callback_list[0]

    for k in H.history.keys():
        if not k.startswith("val_"):
            plt.plot(H.history[k])
            plt.plot(H.history["val_"+k])
            plt.legend(['train', 'val'])
            plt.title(k)
            plt.xlabel('epoch')
            plt.savefig(os.path.join(MODEL_DIR, k+".png"))
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

    for epoch in range(args.epochs):
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

args.name = modelname
# just run the evaluate script
import evaluate
