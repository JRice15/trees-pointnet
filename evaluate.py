"""
This file is mostly a wrapper around LidarPatchGen.evaluate_model(...)
"""

import contextlib
import datetime
import glob
import json
import argparse
import os
from pprint import pprint
from pathlib import PurePath
import time

import h5py
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model
from tensorflow.keras import backend as K
from tensorflow.keras import callbacks, layers
from tensorflow.keras.optimizers import Adam

from core import DATA_DIR, REPO_ROOT, ARGS, data_loading
from core.losses import get_loss
from core.models import pointnet
from core.tf_utils import MyModelCheckpoint, output_model, load_saved_model, generate_predictions
from core.viz_utils import raster_plot

matplotlib.rc_file_defaults()

def errors_plot(pred, y, results_dir):
    x_w = np.empty(pred.shape)
    x_w.fill(1/pred.shape[0])
    y_w = np.empty(y.shape)
    y_w.fill(1/y.shape[0])
    low = int(min(pred.min(), y.min()))
    high = int(max(pred.max(), y.max()))
    step = max((high - low) // 20, 1)
    bins = range(low, high+1, step)
    plt.hist(y, bins=bins, weights=y_w, label="gt", alpha=0.5, color="green")
    plt.hist(pred, bins=bins, weights=x_w, label="predictions", alpha=0.5, color="blue")
    plt.title("Predictions and Ground Truth Values")
    plt.axvline(np.mean(y), label="gt mean", color="green", linestyle="--")
    plt.axvline(np.mean(pred), label="prediction mean", color="blue", linestyle="--")
    plt.legend()
    plt.savefig(os.path.join(results_dir, "preds_vs_gt_hist.png"))
    plt.close()

def count_errors(pred, y):
    if len(pred.shape) > 1:
        pred = K.sum(pred[...,-1], axis=-1)
        y = K.sum(y[...,-1], axis=-1)
    return pred - y


def main():
    """
    Parse arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--name",required=True,help="name of model to run, with possible timestamp in front")
    parser.add_argument("--test",action="store_true",help="run minimal functionality for testing")
    parser.parse_args(namespace=ARGS)


    MODEL_DIR = MODEL_PATH.parent
    EVAL_DIR = MODEL_DIR.joinpath("evaluation")
    os.makedirs(EVAL_DIR, exist_ok=True)

    # load original params into ARGS object
    params_file = MODEL_DIR.joinpath("params.json")
    with open(params_file, "r") as f:
        params = json.load(f)
    params.pop("name")
    params.pop("test")
    for k,v in params.items():
        setattr(ARGS, k, v)

    pprint(vars(ARGS))

    """
    Evaluation
    """

    model = load_saved_model(MODEL_PATH.as_posix(), ARGS)

    test_gen = data_loading.get_test_gen()
    test_gen.summary()

    test_gen.evaluate_model(model, EVAL_DIR)

    # """
    # Mode-specific evaluations
    # """
    # print("Evaluating errors")

    # # error between target and prediction scalar labels
    # if ARGS.mode in ["count"]:
    #     errors_plot(pred, y, EVAL_DIR)

    # # error histogram
    # if ARGS.mode in ["count", "pwtt"]:
    #     errors = count_errors(pred, y)

    #     plt.hist(errors)
    #     plt.title("Errors (pred - gt)")
    #     plt.vlines(np.mean(errors), 0, plt.ylim()[1], label="mean", colors="black")
    #     plt.legend()
    #     plt.savefig(os.path.join(EVAL_DIR, "errors_hist.png"))
    #     plt.close()



if __name__ == "__main__":
    main()
