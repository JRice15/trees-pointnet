import argparse
import os
import json

import optuna

from hpo_utils import get_study, studypath

parser = argparse.ArgumentParser()
parser.add_argument("--name",required=True)
parser.add_argument("--newname",required=True)
parser.add_argument("--no-pruned",action="store_true")
parser.add_argument("--only",nargs="+",default=None,type=int,help="include only these trials")
ARGS = parser.parse_args()


study = get_study(ARGS.name, assume_exists=True)
with open(studypath(ARGS.name, "metadata.json"), "r") as f:
    meta = json.load(f)
    meta["name"] = ARGS.newname

new_study = get_study(ARGS.newname, assume_exists=False)
with open(studypath(ARGS.newname, "metadata.json"), "w") as f:
    json.dump(meta, f)

trials = study.trials
print(len(trials), "orig trials")
trials = [t for t in trials if t.state != optuna.trial.TrialState.RUNNING]
trials = [t for t in trials if t.state != optuna.trial.TrialState.WAITING]
print(len(trials), "after removing running/waiting trials")
if ARGS.only is not None:
    trials = [t for t in trials if t.number in ARGS.only]
    print(len(trials), "after only")
if ARGS.no_pruned:
    trials = [t for t in trials if t.state != optuna.trial.TrialState.PRUNED]
    print(len(trials), "after removing pruned")

new_study.add_trials(trials)

print(new_study.trials_dataframe())

