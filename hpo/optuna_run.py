"""

Creates a three-level system for parallelized hyper-parameter optimization
For example, with 2 GPUs and 2 workers per GPU:

                 master
    ________________|_________________
    |          |          |          |
    |          |          |          |
worker0-0  worker0-1  worker1-0  worker1-1
    |-- GPU0 --|          |-- GPU1 --|
    |          |          |          |
  trial      trial      trial      trial


The master persists, essentially just sitting and waiting for a keyboard interrupt,
or for the study to finish

The master spins up N workers (N = num GPUS * workers per GPU), which spawn trials
repeatedly. The workers also persist until killed by keyboard interrupt, or the study
finishes.

Trials are a subprocess of the worker, which runs the actual training. Trial run until
complete, and then are replaced. They can error out without hurting the study as well.
"""



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
                       ignore_kbint, studypath)

# frequency (seconds) with which workers check their subprocesses
WORKER_POLL_FREQ = 10


def make_objective_func(ARGS, gpu, interrupt_event):
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
        if ARGS.test:
            constant_params["epochs"] = 1
        constant_flags = [
            # don't compute losses or plots
            "noplot",
            "nolosses",
        ]

        # search space params
        params, flags = space.get_params(ARGS, trial)

        # combine the two
        params = dict(**params, **constant_params)
        flags = flags + constant_flags

        cmd = [f"{ROOT}/docker_run.sh", str(gpu), "python", f"{ROOT}/train.py"]
        for name, val in params.items():
            if not isinstance(val, list):
                val = [val]
            val = [str(x) for x in val]
            cmd += ["--"+name] + val
        for flag in flags:
            cmd.append("--"+flag)

        p = subprocess.Popen(cmd)

        start_time = time.perf_counter()
        # wait for process to finish
        while True:
            time.sleep(WORKER_POLL_FREQ)
            poll_val = p.poll()
            # check if we should interrupt
            if interrupt_event.is_set():
                # kill the proc
                p.terminate()
                p.wait()
                raise KilledTrialError()
            # check if process finished
            if poll_val is not None:
                break
            # check if we've timed out
            if (time.perf_counter() - start_time) / 60 > ARGS.timeout_mins:
                raise optuna.TrialPruned("Timeout")
        
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
    optuna callback to stop trials after no improvement for many trials
    """

    def __init__(self, min_trials, stopping_trials, interrupt_event):
        self.min_trials = min_trials
        self.stopping_trials = stopping_trials
        self.interrupt_event = interrupt_event
    
    def __call__(self, study, trial):
        current_trialnum = trial.number
        try:
            best_trialnum = study.best_trial.number
        except ValueError:
            return # no best trial yet exists
        trials_since_best = current_trialnum - best_trialnum
        if best_trialnum >= self.min_trials and trials_since_best >= self.stopping_trials:
            print("NoImprovementStopping triggered.")
            self.interrupt_event.set()


def optuna_worker(ARGS, gpu, worker_num, interrupt_event):
    """
    worker process
    """
    worker_id = "{}-{}".format(gpu, worker_num)
    # study already created by main process
    study = get_study(ARGS.name, assume_exists=True)

    objective = make_objective_func(ARGS, gpu, interrupt_event)

    callbacks = [
        NoImprovementStopping(ARGS.mintrials, ARGS.earlystop, interrupt_event),
    ]

    print("Running worker", worker_id)
    with ignore_kbint():
        # run one step at a time
        while not interrupt_event.is_set():
            try:
                study.optimize(
                    objective,
                    n_trials=1,
                    callbacks=callbacks,
                )
            except MyHpoError:
                pass
            gc.collect()
            # remove core dump files (they can be 20+ gb each)
            for core_file in glob.glob(f"{ROOT}/hpo/core.*"):
                try:
                    os.remove(core_file)
                    print("removed", core_file)
                except FileNotFoundError:
                    pass
                



def main():
    """
    master process
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--name",required=True,help="study name: subdirectory inside models/ to save these trials under")

    # resources
    parser.add_argument("--gpus",required=True,type=int,nargs="+",help="GPU IDs to use")
    parser.add_argument("--per-gpu",type=int,default=1,help="number of concurrent models to train on each GPU")

    # overwrite/resume
    parser.add_argument("--resume",action="store_true",help="resume the study of the given name, otherwise will raise an error if this study already exists")
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

    # flag to signal keyboard interrupt to the workers
    intrpt_event = multiprocessing.Manager().Event()

    procs = []
    for gpu in ARGS.gpus:
        for i in range(ARGS.per_gpu):
            kwargs = {
                "gpu": gpu,
                "worker_num": i,
                "interrupt_event": intrpt_event,
                "ARGS": ARGS,
            }
            p = multiprocessing.Process(target=optuna_worker, kwargs=kwargs)
            procs.append(p)
            p.start()
            # stagger start times
            time.sleep(2)
    
    try:
        for p in procs: 
            p.join()
    except KeyboardInterrupt:
        print("\nStopping processes. This may take up to {} seconds\n".format(WORKER_POLL_FREQ))
        intrpt_event.set()
        for p in procs:
            p.join()
        




if __name__ == "__main__":
    main()
