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

from hpo_utils import get_study




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name",required=True,help="name of study to analyze")
    args = parser.parse_args()

    # load study
    study = get_study(args.name, assume_exists=True)

    df = study.trials_dataframe(["number", "state", "value", "duration", "datetime_start", "datetime_complete"])
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

    optuna.visualization.plot_optimization_history(study).show()
    optuna.visualization.plot_param_importances(study).show()
    optuna.visualization.plot_slice(study).show()


if __name__ == "__main__":
    main()
