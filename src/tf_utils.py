import contextlib
import time
import os
import glob
from pathlib import PurePath

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model
from tensorflow.keras import backend as K
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam

from src import ARGS, DATA_DIR, REPO_ROOT, MODEL_SAVE_FMT, CUSTOM_LAYERS
from src.utils import raster_plot
from src.losses import get_loss


class MyModelCheckpoint(tf.keras.callbacks.ModelCheckpoint):
    """
    https://github.com/tensorflow/tensorflow/issues/33163
    """

    def __init__(self, model_dir, *args, epoch_per_save=1, **kwargs):
        self.epochs_per_save = epoch_per_save
        path = model_dir.joinpath("model" + MODEL_SAVE_FMT).as_posix()
        super().__init__(path, *args, save_freq="epoch", **kwargs)

    def on_epoch_end(self, epoch, logs):
        if epoch % self.epochs_per_save == 0:
            super().on_epoch_end(epoch, logs)

    def best_val_loss(self):
        return self.best


def load_saved_model(model_dir):
    """
    Load the best model iteration saved by ModelCheckpoint for a particular model configuration
    """
    print("Loading model in", model_dir)
    h5path = model_dir.joinpath("model.h5")
    tfpath = model_dir.joinpath("model.tf")
    if os.path.exists(h5path):
        modelpath = h5path
    elif os.path.exists(tfpath):
        modelpath = tfpath
    else:
        raise ValueError("No model found in {}".format(model_dir))

    loss_fun, metrics = get_loss()

    custom_objs = {loss_fun.__name__: loss_fun, **CUSTOM_LAYERS}
    if metrics is not None:
        custom_objs.update({m.__name__:m for m in metrics})

    model = keras.models.load_model(modelpath.as_posix(), custom_objects=custom_objs)

    # model.compile(
    #     optimizer=model.optimizer,
    #     loss=model.loss,
    #     metrics=metrics
    # )
    return model

def output_model(model, directory, show=False):
    """
    print and plot model structure to output dir
    """
    with open(os.path.join(directory, "model_summary.txt"), "w") as f:
        model.summary(print_fn=lambda x: f.write(x + "\n"), line_length=150)
    if show:
        print()
        model.summary()
    try:
        keras.utils.plot_model(model, 
            to_file=os.path.join(directory, "model.png"),
            show_shapes=True,
            #expand_nested=True,
            dpi=192,
        )
    except Exception as e:
        print("Failed to plot model: " + str(e))


