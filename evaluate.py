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
import matplotlib
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
from core.viz_utils import raster_plot

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

    matching_models = [i for i in glob.glob(os.path.join(ALL_MODELS_DIR, ARGS.name+"*", "model_*.tf"))]
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
    loss_fun, metrics = get_loss(ARGS)

    custom_objs = {loss_fun.__name__: loss_fun}
    if metrics is not None:
        custom_objs.update({m.__name__:m for m in metrics})

    model = keras.models.load_model(model_path, custom_objects=custom_objs)

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

    patch_ids = test_gen.ids
    assert len(pred) == len(patch_ids)
    np.savez(os.path.join(EVAL_DIR, "full_predictions.npz"), pred=pred, patch_ids=patch_ids)

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
        os.makedirs(GT_VIS_DIR, exist_ok=True)
        # grab random 10 examples
        for i in range(0, 10*5, 5):
            x, y, patchname = test_gen.get_patch(i)
            ylocs = y[y[...,2] == 1][...,:2]
            raster_plot(ylocs, GT_VIS_DIR+"/{}_gt".format(patchname), gaussian_sigma=ARGS.mmd_sigma)

            if ARGS.mode in ["mmd", "pwtt"]:
                gt_ntrees = len(ylocs)

                x_weights = x[...,-1]
                x_locs = x[...,:2]
                raster_plot(x_locs, gaussian_sigma=ARGS.mmd_sigma, weights=x_weights, sqrt_scale=True, clip=1, 
                    filename=GT_VIS_DIR+"/{}_lidar".format(patchname))

                pred = model.predict(np.expand_dims(x, 0))[0]
                pred_locs = pred[...,:2]
                pred_weights = pred[...,2]
                raster_plot(pred_locs, gaussian_sigma=ARGS.mmd_sigma, weights=pred_weights, 
                    filename=GT_VIS_DIR+"/{}_pred".format(patchname))

                sorted_preds = pred[np.argsort(pred_weights)][::-1]
                topk_locs = sorted_preds[...,:2][:gt_ntrees]
                topk_weights = sorted_preds[...,2][:gt_ntrees]
                raster_plot(topk_locs, gaussian_sigma=ARGS.mmd_sigma, weights=topk_weights, 
                    filename=GT_VIS_DIR+"/{}_pred_topk".format(patchname))


if __name__ == "__main__":
    parse_evaluation_args()
    main()
