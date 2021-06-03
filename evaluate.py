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

    parser.parse_args(namespace=ARGS)


def main():
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
    RESULTS_DIR = os.path.join(MODEL_DIR, "results")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # load original params into ARGS object
    params_file = os.path.join(MODEL_DIR, "params.json")
    with open(params_file, "r") as f:
        params = json.load(f)
    params.pop("name")
    for k,v in params.items():
        setattr(ARGS, k, v)

    pprint(vars(ARGS))


    """
    Setup Testing
    """

    # load best model
    print("Loading model", MODEL_PATH)
    loss_fun, _, error_func = get_loss(ARGS)
    print(loss_fun.__name__)
    model = keras.models.load_model(MODEL_PATH, custom_objects={loss_fun.__name__: loss_fun})

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

    test_gen = data_loading.get_test_gen()
    test_gen.summary()


    """
    Evaluate model
    """

    print()
    x, y = test_gen.load_all()
    y = np.squeeze(y.numpy())
    pred = np.squeeze(model.predict(x))

    print("First 10 predictions, ground truths:")
    for i in range(min(10, len(pred))):
        print("pred {}:".format(i))
        print(pred[i])
        print("gt {}:".format(i))
        print(y[i])


    metric_vals = model.evaluate(test_gen)

    results = {model.metrics_names[i]:v for i,v in enumerate(metric_vals)}

    print() # newline
    for k,v in results.items():
        print(k+":", v)

    with open(os.path.join(RESULTS_DIR, "results.json"), "w") as f:
        json.dump(results, f, indent=2)


    if len(pred.shape) == 1: # if the model outputs a single number
        x_w = np.empty(pred.shape)
        x_w.fill(1/pred.shape[0])
        y_w = np.empty(y.shape)
        y_w.fill(1/y.shape[0])
        low = int(min(pred.min(), y.min()))
        high = int(max(pred.max(), y.max()))
        step = (high - low) // 20
        bins = range(low, high+1, step)
        plt.hist(y, bins=bins, weights=y_w, label="gt", alpha=0.5, color="green")
        plt.hist(pred, bins=bins, weights=x_w, label="predictions", alpha=0.5, color="blue")
        plt.title("Predictions and Ground Truth Values")
        plt.axvline(np.mean(y), label="gt mean", color="green", linestyle="--")
        plt.axvline(np.mean(pred), label="prediction mean", color="blue", linestyle="--")
        plt.legend()
        plt.savefig(os.path.join(RESULTS_DIR, "preds_vs_gt_hist.png"))
        plt.close()

    if error_func is not None:
        errors = error_func(pred, y)

        plt.hist(errors)
        plt.title("Errors (gt - pred)")
        plt.vlines(np.mean(errors), 0, plt.ylim()[1], label="mean", colors="black")
        plt.legend()
        plt.savefig(os.path.join(RESULTS_DIR, "errors_hist.png"))
        plt.close()


if __name__ == "__main__":
    parse_evaluation_args()
    main()
