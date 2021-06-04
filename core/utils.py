import contextlib
import time
import os

import h5py
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model
from tensorflow.keras import backend as K
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam

import matplotlib
import matplotlib.pyplot as plt
import seaborn

from core import ARGS, DATA_DIR

seaborn.set()


class MyModelCheckpoint(tf.keras.callbacks.ModelCheckpoint):
    """
    https://github.com/tensorflow/tensorflow/issues/33163
    """

    def __init__(self, *args, epoch_per_save=1, **kwargs):
        self.epochs_per_save = epoch_per_save
        super().__init__(*args, save_freq="epoch", **kwargs)

    def on_epoch_end(self, epoch, logs):
        if epoch % self.epochs_per_save == 0:
            super().on_epoch_end(epoch, logs)



def output_model(model, directory):
    """
    print and plot model structure to output dir
    """
    with open(os.path.join(directory, "model_summary.txt"), "w") as f:
        model.summary(print_fn=lambda x: f.write(x + "\n"))
    try:
        tf.keras.utils.plot_model(model, to_file=os.path.join(directory, "model.png"))
    except Exception as e:
        print("Failed to plot model: " + str(e))


