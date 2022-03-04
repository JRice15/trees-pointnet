import json
import os
import re
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

from hpo_utils import ROOT, get_study, ignore_kbint, KilledTrial

# frequency with which workers check their subprocesses
WORKER_POLL_FREQ = 2


valid_optimizers = ["adam", "adadelta", "nadam", "adamax"]
valid_output_modes = ["seg", "dense"]
valid_losses = ["mmd", "gridmse"]


def make_objective_func(args, interrupt_event):
    # def objective(trial):
        # train_args = {}

    def objective(trial):
        p = subprocess.Popen(["python3", f"{ROOT}/hpo/temp.py"])
        # wait for process to finish
        while True:
            # poll, then sleep, so that interrupt event has sufficient time to propogate, 
            #   becuase the interrupt kills the subprocess too
            poll_val = p.poll()
            time.sleep(WORKER_POLL_FREQ)
            # check if we should interrupt
            if interrupt_event.is_set():
                # kill the proc
                p.terminate()
                p.wait()
                raise KilledTrial()
            # check if process finished
            if poll_val is not None:
                break

        return trial.suggest_float("x", -10, 10)

    return objective


class NoImprovementStopping:
    """
    optuna callback to stop trials after no improvement for many trials
    """

    def __init__(self, min_trials, no_improvement_trials):
        self.min_trials = min_trials
        self.no_improvement_trials = no_improvement_trials
    
    def __call__(self, study, trial):
        ...


def optuna_worker(worker_id, args, interrupt_event):
    # study already created by main process
    study = get_study(args.name, assume_exists=True)

    objective = make_objective_func(args, interrupt_event)

    print("Running worker", worker_id)
    with ignore_kbint():
        while not interrupt_event.is_set():
            try:
                study.optimize(objective, n_trials=1)
            except KilledTrial:
                pass



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name",required=True,help="subdirectory inside models/ to save these trials under")
    parser.add_argument("--timeout",default=4,type=float,help="max number of hours each model is allowed to run")
    parser.add_argument("--earlystop",type=int,default=50,help="number of trials with no improvement when earlystopping occurs")
    parser.add_argument("--mintrials",type=int,default=200,help="number of trials for which earlystopping is not allowed to occur")
    parser.add_argument("--resume",action="store_true",help="resume the study of the given name, otherwise will raise an error if this study already exists")
    parser.add_argument("--overwrite",action="store_true",help="overwrite study of the given name if it exists")
    # environment
    # parser.add_argument("--envname",required=True,help="name of conda environment")
    parser.add_argument("--gpu-ids","--gpus",type=int,nargs="+",required=True,help="GPU IDs to use")
    parser.add_argument("--per-gpu",type=int,default=2,help="number of concurrent models to train on each GPU")
    args = parser.parse_args()

    assert not (args.overwrite and args.resume)
    if args.overwrite:
        dbpath = f"{ROOT}/hpo/studies/{args.name}.db"
        if os.path.exists(dbpath):
            os.remove(dbpath)

    # create the study, or verify existence
    get_study(args.name, assume_exists=args.resume)

    # flag to signal keyboard interrupt to the workers
    intrpt_event = multiprocessing.Manager().Event()

    procs = []
    for gpu in args.gpu_ids:
        for i in range(args.per_gpu):
            worker_id = "{}-{}".format(gpu, i)
            p = multiprocessing.Process(target=optuna_worker, args=(worker_id, args, intrpt_event))
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
