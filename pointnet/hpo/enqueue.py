import optuna
import re
import json
import argparse

from hpo_utils import studypath, get_study

parser = argparse.ArgumentParser()
parser.add_argument("--name",required=True)
parser.add_argument("--basetrial",required=True,type=int)
parser.add_argument("--count",default=1,type=int)

parsed, unknown = parser.parse_known_args()
for name in unknown:
    if name.startswith("--"):
        kwargs = {}
        if name == "--channels":
            kwargs["nargs"] = "+"
        parser.add_argument(name, **kwargs)
ARGS = parser.parse_args()

study = get_study(ARGS.name, assume_exists=True)

params = study.trials[ARGS.basetrial].params

print("orig:")
print(params)

def interp_type(x):
    try:
        return int(x)
    except ValueError:
        pass
    try:
        return float(x)
    except ValueError:
        return x

for key, value in vars(ARGS).items():
    if key in ("name", "basetrial", "count"):
        continue
    key = re.sub(r"_", "-", key)
    if not isinstance(value, list):
        try:
            dtype = type(params[key])
            value = dtype(value)
        except KeyError:
            value = interp_type(value)
            print("interpolated type for", key, "to", type(value))
    params[key] = value

print(params)

print(f"enqueue {ARGS.count} times? y/[n]:", end=" ")
if input().startswith("y"):
    for _ in range(ARGS.count):
        study.enqueue_trial(params)
    print("done")
else:
    print("cancelled")

study.trials[-1].set_user_attr("basetrial", ARGS.basetrial)
