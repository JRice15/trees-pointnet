import json
import os
import re
import time
import glob
import argparse
import pickle
import shutil
import sys
from pathlib import PurePath
import time
from pprint import pprint, pformat
import subprocess

import numpy as np
import pandas as pd
import hyperopt
from hyperopt import hp, STATUS_OK, STATUS_FAIL

from src import ARGS

# TODO select a GPU
# template sbatch script
program_template = """
echo 'log start' &> {log}
echo '{name} {model}' &>> {log}
date &>> {log}
source ~/.bashrc &>> {log}
conda activate {envname} &>> {log}
python3 train.py {train_args} &>> {log}
python3 pts_to_csv.py {train_args} &>> {log}
"""
program_template = re.sub(r"\n", "; ", program_template.strip())


def build_search_space():
    # define hyperopt search space
    search_space = {
        "optimizer": hp.choice("opt", ["adam", "nadam", "adadelta", "adamax"]),
        "batchsize": 2 ** hp.quniform("batchsize_exp", 4, 8, 1), # min 16, max 256
        "lr": 10 ** hp.quniform("lr", -5, -1, 0.5), # 0.00001 to 0.1
        "reducelr-factor": hp.quniform("lr-factor", 0.1, 1.0, 0.1), # ReduceLROnPlateau
        "reducelr-patience": hp.quniform("lr-patience", 5, 50, 5), # ReduceLROnPlateau
        "npoints": hp.quniform("npoints", 200, 1500, 100),
        # "dropout": hp.choice("use-dropout", [
        #     False,
        #     hp.quniform("dropout", 0.05, 0.8, 0.05)
        # ]),
        "ndvi": hp.choice("ndvi", [False, True]),
        "gaussian-sigma": ARGS.abs_sigma,
        "mmd-kernel": ARGS.mmd_kernel,
        "ortho-weight": 10 ** hp.quniform("ortho", -6, 0, 0.5) # 1e-6 to 1
    }


    return search_space


def make_objective_func(trialnum):
    """
    wrapper for hyperopt objective function, to take in metadata args
    """
    def objective_func(space):
        print(space)
        # parse arguments
        name = "hp_{}_{}".format(ARGS.mode, trialnum)
        log = "hp/logs/" + name + ".log"

        # remove previously run models
        matches = sorted(glob.glob("models/{}-{}*/".format(name, ARGS.mode)))
        for match in matches:
            shutil.rmtree(match)

        # train args
        train_args = ["--name", name, "--mode", ARGS.mode]
        space["batchsize"] = int(space["batchsize"])
        space["npoints"] = int(space["npoints"])
        for k,v in space.items():
            if v is False or v is None:
                continue
            elif v is True:
                train_args += ["--"+k]
            else:
                train_args += ["--"+k, str(v)]
        # pts 2 csv args
        pts2csv_args = [
            "--name", name+"-"+ARGS.mode, # qualified name
            "--threshold", 0.02
        ]


        global program_template
        program = program_template.format(name=name, model=ARGS.model, log=log,
                                    envname=ARGS.envname,
                                    train_args=" ".join(train_args))
        
        print("Beginning subprocess")
        try:
            subprocess.run(program, shell=True, check=True,
                           timeout=(60 * 60 * ARGS.timeout) # timeout in seconds
                        )
        except subprocess.CalledProcessError as e:
            print("! Error:", str(e))
            return {"status": STATUS_FAIL, "loss": None}
        print("Subprocess complete")

        paths = glob.glob("models/{}-{}-*/".format(name, ARGS.mode))
        path = sorted(paths)[-1] # get most recent if multiple
        modeldir = PurePath(path)

        # training resulted in error somehow
        if os.path.exists(modeldir.joinpath("training_failed.txt")):
            print("Training failed:")
            with open(modeldir.joinpath("training_failed.txt"), "r") as f:
                print(f.read())
            shutil.rmtree(modeldir)
            return {"status": STATUS_FAIL, "loss": None}
        # training finished
        elif os.path.exists(modeldir.joinpath("evaluation/results.json")):
            with open(modeldir.joinpath("evaluation/results.json"), "r") as f:
                results = json.load(f)
            # we don't actually care about the testing loss, as it may have regularization terms
            #  and such that we only care about minimizing during training. If a model does poorly
            #  on those but still achieves the best MSE, we want that model (and also to change our regularization)
            results["test_loss"] = results["loss"]
            results["loss"] = results["mse"]
            results["status"] = STATUS_OK
            print(name, "results:", results)
            return results
        # otherwise training timed out
        else:
            print("Training did not succeed within the {} hr time limit".format(ARGS.timeout))
            return {"status": STATUS_FAIL, "loss": None}

    return objective_func



def main():
    parser = argparse.ArgumentParser()
    # general args
    parser.add_argument("--trials-name","--name",required=True,help="name to save these trials under")
    parser.add_argument("--output-mode",required=True)
    parser.add_argument("--loss",required=True)
    parser.add_argument("--timeout",default=4,type=float,help="max number of hours each model is allowed to run")
    parser.add_argument("--earlystop",type=int,default=50,help="number of trials with no improvement when earlystopping occurs")
    parser.add_argument("--mintrials",type=int,default=200,help="number of trials for which earlystopping is not allowed to occur")
    # environment
    parser.add_argument("--envname",help="name of conda environment")
    parser.add_argument("--gpu",type=int)
    # mode specific arguments
    parser.add_argument("--gaussian-sigma",default=None,type=float)
    parser.add_argument("--mmd-kernel",default=None)
    # alternate functionality flags
    parser.add_argument("--show-best",action="store_true",help="instead of running hyperopt, show the best model found so far")

    parser.parse_args(namespace=ARGS)

    # only support this mode at the moment
    assert ARGS.mode == "pwmmd"

    if "mmd" in ARGS.mode:
        assert ARGS.abs_sigma is not None
        assert ARGS.mmd_kernel is not None

    RUN_NAME = ARGS.trials_name + "_" + ARGS.mode
    TRIALS_FILE = "hp/{}/trials.pickle".format(RUN_NAME)
    META_FILE = "hp/{}/meta.json".format(RUN_NAME)
    metadata = {
        "gaussian-sigma": ARGS.abs_sigma,
        "mmd-kernel": ARGS.mmd_kernel,
        "timeout": ARGS.timeout,
    }

    search_space = build_search_space()

    # create or load hyperopt.Trials object
    if os.path.exists(TRIALS_FILE):
        with open(TRIALS_FILE, "rb") as f:
            trials = pickle.load(f)
        # showing best trial, then exit
        if ARGS.show_best:
            print("Best trial:", trials.best_trial["tid"])
            pprint(trials.best_trial["result"])
            pprint(
                    hyperopt.space_eval(search_space, {k:v[0] for k,v in trials.best_trial["misc"]["vals"].items()})
            )
            exit()
        # make sure meta arguments match
        with open(META_FILE, "r") as f:
            saved_meta = json.load(f)
        if saved_meta != metadata:
            raise ValueError("Meta arguments did not match saved data: {} != {}".format(saved_meta, metadata))
        print("Loaded trials, {} trials already completed".format(len(trials.trials)))
    else:
        if ARGS.show_best:
            raise ValueError("Cannot show best trial, no matching trials found!")
        trials = hyperopt.Trials()
        print("Creating new trials")

        # save metadata
        os.makedirs("hp/" + RUN_NAME)
        with open(META_FILE, "w") as f:
            json.dump(metadata, f)

    assert ARGS.envname is not None
    assert ARGS.gpu is not None

    while True:
        # do one trial
        hyperopt.fmin(
            make_objective_func(trialnum=len(trials.trials)),
            algo=hyperopt.tpe.suggest,
            space=search_space,
            max_evals=len(trials.trials)+1,
            trials=trials,
        )

        # save trials to pickle
        with open(TRIALS_FILE, "wb") as f:
            pickle.dump(trials, f)

        if ARGS.earlystop:
            if len(trials.trials) >= ARGS.min_trials:
                if len(trials.trials) - trials.best_trial["tid"] >= ARGS.earlystop:
                    print("Manual early stopping of Hyperoptimization: no improvement for {} iterations".format(ARGS.earlystop))
                    exit()

if __name__ == "__main__":
    main()
