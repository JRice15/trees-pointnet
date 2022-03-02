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





def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name",required=True,help="subdirectory inside models/ to save these trials under")

    args = parser.parse_args()

    # load study
    storage = "sqlite:///hpo/{}.db".format(args.name)
    try:
        study = optuna.load_study(
            study_name=args.name,
            storage=storage
        )
    except Exception as e:
        raise ValueError(f"Error loading study '{args.name}': {str(e)}")

    df = study.trials_dataframe()

    print(df.tail(25))

    print(df["datetime_complete"] - df["datetime_start"])

    # print(study.trials)
    print(study.best_trial)




if __name__ == "__main__":
    main()
