import contextlib
import datetime
import glob
import json
import argparse
import os
from pprint import pprint
import time

import h5py
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model
from tensorflow.keras import backend as K
from tensorflow.keras import callbacks, layers
from tensorflow.keras.optimizers import Adam
import seaborn

from core import DATA_DIR, MAIN_DIR, ARGS, data_loading
from core.losses import get_loss
from core.models import pointnet
from core.utils import MyModelCheckpoint, output_model

seaborn.set()

"""
parse args
"""

def parse_evaluation_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name",required=True,help="name of model to run, with possible timestamp in front")
    parser.add_argument("--test",action="store_true",help="run minimal functionality for testing")
    parser.parse_args(namespace=ARGS)

def init_evaluation_args():
    ALL_MODELS_DIR = os.path.join(MAIN_DIR, "models/")

    matching_models = [i for i in glob.glob(os.path.join(ALL_MODELS_DIR, "*"+ARGS.name, "model_*.tf"))]
    if len(matching_models) > 1:
        print("Multiple models match 'name' argument:")
        print(" ", matching_models)
        print("Defaulting to the most recent:")
        # all the names begin with a date/time string, so sorting gives order by time
        matching_models.sort()
        MODEL_PATH = matching_models[-1]
        print(" ", MODEL_PATH)
        print("You can add the timestamp string to '--name' to specify a different model")
    elif len(matching_models) == 0:
        print("No matching models!")
        exit()
    else:
        MODEL_PATH = matching_models[0]
    MODEL_DIR = os.path.dirname(MODEL_PATH)
    EVAL_DIR = os.path.join(MODEL_DIR, "evaluation")
    os.makedirs(EVAL_DIR, exist_ok=True)

    # load original params into ARGS object
    params_file = os.path.join(MODEL_DIR, "params.json")
    with open(params_file, "r") as f:
        params = json.load(f)
    params.pop("name")
    params.pop("test")
    for k,v in params.items():
        setattr(ARGS, k, v)

    pprint(vars(ARGS))

    return MODEL_PATH, MODEL_DIR, EVAL_DIR

def load_model(model_path, ARGS):
    print("Loading model", model_path)
    loss_fun, _ = get_loss(ARGS)

    model = keras.models.load_model(model_path, custom_objects={loss_fun.__name__: loss_fun})

    metrics = None
    # additional metrics not used in training
    if ARGS.mode == "count":
        metrics = [
            "mean_squared_error",
            "mean_absolute_error",
            "RootMeanSquaredError",
            "mean_absolute_percentage_error",
        ]

    model.compile(
        optimizer=model.optimizer,
        loss=model.loss,
        metrics=metrics
    )
    return model


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

def gaussian(x, center, sigma=0.05):
    const = (2 * np.pi * sigma) ** -0.5
    exp = np.exp( -np.sum((x - center) ** 2, axis=-1) / (2 * sigma ** 2))
    return const * exp

def raster_plot(pts, name, weights=None, title=None):
    x = np.linspace(0, 1)
    y = np.linspace(0, 1)
    x, y = np.meshgrid(x, y)
    gridpts = np.stack([x,y], axis=-1)
    gridvals = np.zeros_like(x)
    for i,p in enumerate(pts):
        vals = gaussian(gridpts, p)
        if weights is not None:
            vals *= max(weights[i], 0)
        cutoff = 2 * np.median(vals)
        vals = np.where(vals > cutoff, 
            ((vals - cutoff) * 0.1) + cutoff,
            vals)
        gridvals += vals

    plt.pcolormesh(x,y,gridvals, shading="auto")
    plt.colorbar()
    if title is not None:
        plt.title(title)
    plt.tight_layout()
    plt.savefig(name)
    plt.close()

def count_errors(pred, y):
    if len(pred.shape) > 1:
        pred = K.sum(pred[...,-1], axis=-1)
        y = K.sum(y[...,-1], axis=-1)
    return pred - y


def main():
    MODEL_PATH, MODEL_DIR, EVAL_DIR = init_evaluation_args()

    """
    Setup Testing
    """
    model = load_model(MODEL_PATH, ARGS)

    test_gen = data_loading.get_test_gen()
    test_gen.summary()

    """
    Raw predictions
    """
    print("Generating Raw Predictions")

    x, y = test_gen.load_all()
    y = np.squeeze(y.numpy())
    pred = np.squeeze(model.predict(x))

    with open(os.path.join(EVAL_DIR, "sample_predictions.txt"), "w") as f:
        f.write("First 10 predictions, ground truths:\n")
        for i in range(min(10, len(pred))):
            f.write("pred {}:\n".format(i))
            f.write(str(pred[i])+"\n")
            f.write("gt {}:\n".format(i))
            f.write(str(y[i])+"\n")

    """
    Evaluate Metrics
    """
    print("Evaluating metrics")

    metric_vals = model.evaluate(test_gen)
    results = {model.metrics_names[i]:v for i,v in enumerate(metric_vals)}

    print() # newline
    for k,v in results.items():
        print(k+":", v)

    with open(os.path.join(EVAL_DIR, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    """
    Mode-specific evaluations
    """
    print("Evaluating errors")

    # error between target and prediction scalar labels
    if ARGS.mode in ["count"]:
        errors_plot(pred, y, EVAL_DIR)

    # error histogram
    if ARGS.mode in ["count", "pwtt"]:
        errors = count_errors(pred, y)

        plt.hist(errors)
        plt.title("Errors (pred - gt)")
        plt.vlines(np.mean(errors), 0, plt.ylim()[1], label="mean", colors="black")
        plt.legend()
        plt.savefig(os.path.join(EVAL_DIR, "errors_hist.png"))
        plt.close()
    
    """
    Visualizations
    """
    print("Generating visualizations")

    # data visualizations
    if ARGS.mode not in ["count"]:
        test_gen.sorted()
        GT_VIS_DIR = os.path.join(EVAL_DIR, "visualizations")
        os.makedirs(GT_VIS_DIR+"/lidar", exist_ok=True)
        os.makedirs(GT_VIS_DIR+"/gt", exist_ok=True)
        os.makedirs(GT_VIS_DIR+"/predictions", exist_ok=True)
        # grab one example from ~20 batches
        for i in range(0, len(test_gen), len(test_gen)//20):
            full_x, full_y = test_gen[i]
            x = full_x[0]
            y = full_y[0]
            y = y[y[...,2] == 1]
            raster_plot(y[...,:2], GT_VIS_DIR+"/gt/gt{}".format(i))
            if ARGS.mode in ["mmd", "pwtt"]:
                x_weights = x[...,-1]
                x_locs = x[...,:2]
                raster_plot(x_locs, weights=x_weights, name=GT_VIS_DIR+"/lidar/lidar{}".format(i))

                pred = model.predict(full_x)[0]
                pred_locs = pred[...,:2]
                pred_weights = pred[...,2]
                raster_plot(pred_locs, weights=pred_weights, name=GT_VIS_DIR+"/predictions/pred{}".format(i))



if __name__ == "__main__":
    parse_evaluation_args()
    main()
