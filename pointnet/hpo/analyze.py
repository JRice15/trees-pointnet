import json
import os
import re
import time
import glob
import argparse
import pickle
import shutil
import sys
import subprocess
import multiprocessing
from pathlib import PurePath
from pprint import pprint, pformat

import numpy as np
import pandas as pd
import optuna

from hpo_utils import get_study, studypath



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name",required=True,help="name of study to analyze")
    args = parser.parse_args()

    # load study
    study = get_study(args.name, assume_exists=True)

    print("total time:", study.trials_dataframe(["duration"]).sum())

    df = study.trials_dataframe(["number", "state", "value", "duration", "user_attrs", "system_attrs"])
    print("All trials:")
    print(df)

    print("\nRecent Trials:")
    print(df.tail(20))

    print("\nBest trials:")
    print(df.sort_values(by="value", ascending=False).head(15))

    print("\nBest trial:")
    print("  Value:", study.best_value)
    print("  Best Params:", study.best_params)

    pdf = study.trials_dataframe(["params"])
    print("\n10 most recent params:")
    print(pdf.tail(10).T)

    print("\nBest Params:")
    pdf = study.trials_dataframe(["value", "params"])
    print(pdf.sort_values(by="value", ascending=False).head(10).T)

    running = (df["state"] == "RUNNING").sum()
    print(running, "trials are still running:")
    print(df[df["state"] == "RUNNING"])

    print("plotting")
    optuna.visualization.plot_optimization_history(study) \
        .write_image(studypath(args.name, "optimization_history.png"), scale=2)
    optuna.visualization.plot_param_importances(study) \
        .write_image(studypath(args.name, "param_importances.png"), scale=2)
    optuna.visualization.plot_slice(study) \
        .write_image(studypath(args.name, "slice_plot.png"), scale=2)



if __name__ == "__main__":
    main()
