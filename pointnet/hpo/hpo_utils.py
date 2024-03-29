import json
import os
import time
import glob
import argparse
import signal
import logging
import sys
import subprocess
import multiprocessing
from pathlib import PurePath

import numpy as np
import pandas as pd
import optuna

from os.path import dirname as dirn
# root of repository
#   up three levels: hpo -> pointnet -> trees-pointnet
ROOT = dirn(dirn(dirn(os.path.abspath(__file__))))

class MyHpoError(Exception): pass
class TrialTimeoutError(MyHpoError): pass
class KilledTrialError(MyHpoError): pass
class TrialFailedError(MyHpoError): pass


def studypath(study_name, filename=None):
    if filename is None:
        return f"{ROOT}/pointnet/hpo/studies/{study_name}/"
    return f"{ROOT}/pointnet/hpo/studies/{study_name}/{filename}"

def get_study(name, assume_exists):
    """
    get optuna study. If actual existence is different from assume_exists, error is raised
    args:
        name: str
        assume_exists: bool
    """
    if assume_exists:
        if not os.path.exists(studypath(name, "study.db")):
            raise ValueError(f"study {name} does not exist")

    os.makedirs(studypath(name), exist_ok=True)
    heartbeat = 5*60 # 5 minutes, in seconds
    storage = optuna.storages.RDBStorage(
        url="sqlite:///" + studypath(name, "study.db"),
        engine_kwargs={"connect_args": {"timeout": 15}}, # longer than usual timeout is ok, because 15s is insignificant compared to the time each trial takes
        heartbeat_interval=heartbeat,
        grace_period=3*heartbeat, # trials fail after 15 minutes of inactivity
    )
    sampler = optuna.samplers.TPESampler(
        multivariate=True, # consider the relations between different parameters
        group=True,
        warn_independent_sampling=True,
        constant_liar=True, # avoid very similar param combinations being tried simultaneously
        n_startup_trials=30, # number of random sample trials to begin with
        seed=1234,
    )

    if assume_exists:
        try:
            study = optuna.load_study(
                study_name=None,
                storage=storage,
                sampler=sampler,
            )
        except Exception as e:
            raise ValueError(f"Error loading study '{name}': {str(e)}")
    else:
        try:
            study = optuna.create_study(
                study_name=name,
                storage=storage,
                direction="maximize",
                sampler=sampler,
            )
        except optuna.exceptions.DuplicatedStudyError:
            raise ValueError(f"Study named '{name}' already exists. Supply the --resume flag if you want to resume it")

    optuna.storages.fail_stale_trials(study)
    return study

def get_last_trial_status(study):
    series = study.trials_dataframe(["state"])["state"]
    series = series[series != "RUNNING"]
    return series.iloc[-1]

def glob_modeldir(modelname):
    glob_path = f"{ROOT}/pointnet/models/{modelname}-??????-??????"

    matching_models = glob.glob(glob_path)

    if len(matching_models) == 0:
        raise FileNotFoundError("No matching models!")

    if len(matching_models) > 1:
        print("Multiple models match 'name' argument:")
        print(" ", matching_models)
        print("Defaulting to the most recent:")
        # all the names have date/time string, so sorting gives order by time
        matching_models.sort()

    model_dir = matching_models[-1]
    print(" ", model_dir)

    return model_dir
