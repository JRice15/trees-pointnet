import shutil
import os
import argparse

from hpo_utils import ROOT

parser = argparse.ArgumentParser()
parser.add_argument("--name", required=True)
ARGS = parser.parse_args()

os.remove(f"{ROOT}/hpo/studies/{ARGS.name}.db")
os.remove(f"{ROOT}/hpo/studies/{ARGS.name}.json")

shutil.rmtree(f"{ROOT}/models/hpo/study-{ARGS.name}")
