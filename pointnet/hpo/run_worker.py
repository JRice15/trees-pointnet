import argparse
import gc
import glob
import json
import logging
import multiprocessing
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import PurePath

import numpy as np
import optuna
import pandas as pd

import search_spaces
from hpo_utils import (ROOT, KilledTrialError, MyHpoError, TrialFailedError,
                       TrialTimeoutError, get_study, glob_modeldir,
                       studypath)

# frequency (seconds) with which workers check their subprocesses
WORKER_POLL_FREQ = 10


def make_objective_func(ARGS):
    # get search space
    space = getattr(search_spaces, ARGS.search_space)
    assert issubclass(space, search_spaces.SearchSpace)

    def objective(trial):
        """
        function that spawns a trial process
        """
        # constants/meta params
        constant_params = {
            "name": "hpo/{name}/{name}_trial{number}".format(name=ARGS.name, number=trial.number),
            "dsname": ARGS.dsname,
            "eval-sets": ["val", "test"]
        }
        constant_flags = [
            # don't compute losses or plots
            "noplot",
            "nolosses",
        ]
        if ARGS.test:
            constant_flags.append("test")

        # search space params
        params, flags = space.get_params(ARGS, trial)

        # combine the two
        params = dict(**params, **constant_params)
        flags = flags + constant_flags

        cmd = [f"{ROOT}/pointnet/docker_run.sh", str(ARGS.gpu), "python", f"{ROOT}/pointnet/train.py"]
        for name, val in params.items():
            if not isinstance(val, list):
                val = [val]
            val = [str(x) for x in val]
            cmd += ["--"+name] + val
        for flag in flags:
            cmd.append("--"+flag)

        p = subprocess.Popen(cmd)
        start_time = time.perf_counter()

        try:
            # wait for process to finish
            while True:
                time.sleep(WORKER_POLL_FREQ)
                # check if process finished
                if p.poll() is not None:
                    break
                # check if we've timed out
                if (time.perf_counter() - start_time) / 60 > ARGS.timeout_mins:
                    raise optuna.TrialPruned("Timeout")
        except KeyboardInterrupt:
            # make sure subprocess dies
            p.terminate()
            p.wait()
            raise
        
        # try to read results
        try:
            model_dir = glob_modeldir(params["name"])
        except FileNotFoundError:
            raise optuna.TrialPruned("Model dir does not exist")
        results_file = model_dir + "/results_validation/results_pointmatch.json"
        if not os.path.exists(results_file):
            raise optuna.TrialPruned(f"Results file does not exist:\n{results_file}")

        with open(results_file, "r") as f:
            results = json.load(f)
        
        return results["metrics"]["fscore"]

    return objective


class NoImprovementStopping:
    """
    stop optimization after no improvement for many trials
    """

    def __init__(self, study, min_trials, stopping_trials):
        self.study = study
        self.min_trials = min_trials
        self.stopping_trials = stopping_trials
    
    def should_stop(self):
        current_trialnum = self.study.trials[-1].number
        # check min trials
        if current_trialnum < self.min_trials:
            return False
        # get num since best trial
        try:
            best_trialnum = self.study.best_trial.number
        except ValueError:
            return False # no best trial yet exists
        trials_since_best = current_trialnum - best_trialnum
        return trials_since_best > self.stopping_trials


def optuna_worker(ARGS, study):
    """
    worker process
    """
    objective = make_objective_func(ARGS)
    condition = NoImprovementStopping(study, ARGS.mintrials, ARGS.earlystop)

    # run one step at a time
    while True:
        try:
            study.optimize(
                objective,
                n_trials=1,
            )
        except MyHpoError as e:
            print(e.__class__.__name__, e)
        gc.collect()
        # remove core dump files (they can be 20+ gb each)
        for core_file in glob.glob(f"{ROOT}/pointnet/hpo/core.*"):
            try:
                os.remove(core_file)
                print("removed", core_file)
            except FileNotFoundError:
                pass

        # check stopping
        if condition.should_stop():
            return

                



def main():
    """
    master process
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--name",required=True,help="study name: subdirectory inside models/ to save these trials under")
    parser.add_argument("--gpu",required=True,type=int,help="GPU ID to use")

    # overwrite/resume
    parser.add_argument("--resume",action="store_true",help="resume (or run another workr for) the study of the given name, otherwise will raise an error if this study already exists")
    parser.add_argument("--overwrite",action="store_true",help="overwrite study of the given name if it exists")

    # searching params
    parser.add_argument("--search-space",help="(if not resuming:) name of the search space function (defined in search_spaces.py) to use")
    parser.add_argument("--dsname",help="(if not resuming:) name of dataset to use ")
    parser.add_argument("--earlystop",type=int,default=50,help="number of trials with no improvement when earlystopping occurs")
    parser.add_argument("--mintrials",type=int,default=200,help="number of trials for which earlystopping is not allowed to occur")
    parser.add_argument("--timeout-mins",type=float,default=(60*5),help="timeout for individual trials, in minutes (default 5 hours)")

    # misc
    parser.add_argument("--test",action="store_true",help="just run one-epoch trials")
    ARGS = parser.parse_args()

    if "-" in ARGS.name:
        raise ValueError("Use underscores in name instead of dashes")

    assert not (ARGS.overwrite and ARGS.resume), "Cannot both overwrite and resume!"
    if ARGS.overwrite:
        if os.path.exists(studypath(ARGS.name)):
            os.remove(study_path)

    if ARGS.resume:
        # read metadata
        with open(studypath(ARGS.name, "metadata.json"), "r") as f:
            params = json.load(f)
        # allow supplying of dsname/search-space as long as they match meta
        if ARGS.search_space is not None:
            assert ARGS.search_space == params["search_space"], "search_space does not match saved metadata value when using `--resume` flag"
        if ARGS.dsname is not None:
            assert ARGS.dsname == params["dsname"], "dsname does not match saved metadata value when using `--resume` flag"
        ARGS.search_space = params["search_space"]
        ARGS.dsname = params["dsname"]

    space = getattr(search_spaces, ARGS.search_space)
    if not issubclass(space, search_spaces.SearchSpace):
        raise ValueError("Invalid search space: '{}', type {}".format(ARGS.search_space, space))
    if ARGS.dsname is None:
        raise ValueError("Dataset name is required")
    # save metadata
    os.makedirs(studypath(ARGS.name), exist_ok=True)
    with open(studypath(ARGS.name, "metadata.json"), "w") as f:
        json.dump(vars(ARGS), f)

    # create the study, or verify existence
    study = get_study(ARGS.name, assume_exists=ARGS.resume)
    # add default trial
    if len(study.trials) == 0:
        try:
            default_params = space.defaults
        except AttributeError:
            raise ValueError("No default params exist for this study")
        print("Enqueueing default trial")
        study.enqueue_trial(default_params)

    optuna_worker(ARGS, study)





if __name__ == "__main__":
    main()
