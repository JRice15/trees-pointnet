import optuna
import argparse

from hpo_utils import get_study

parser = argparse.ArgumentParser()
parser.add_argument("--name",required=True)
parser.add_argument("--nums",nargs="+",required=True,type=int)
parser.add_argument("--state",default="FAIL")
ARGS = parser.parse_args()

study = get_study(ARGS.name, assume_exists=True)

new_state = getattr(optuna.trial.TrialState, ARGS.state)
for num in ARGS.nums:
    trial = study.trials[num]
    assert trial.number == num
    trial_id = trial._trial_id
    print(trial.number, trial_id)

    print(study._storage.set_trial_state(trial_id, new_state))
