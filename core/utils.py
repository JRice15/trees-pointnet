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

from core import args, OUTPUT_DIR, DATA_DIR


def output_model(model):
    """
    print and plot model structure to output dir
    """
    with open(os.path.join(OUTPUT_DIR, "model_summary.txt"), "w") as f:
        model.summary(print_fn=lambda x: f.write(x + "\n"))
    try:
        tf.keras.utils.plot_model(model, to_file=os.path.join(OUTPUT_DIR, "model.png"))
    except Exception as e:
        print("Failed to plot model: " + str(e))
