import shutil
import os
import argparse

from hpo_utils import ROOT, studypath

parser = argparse.ArgumentParser()
parser.add_argument("--name", required=True)
ARGS = parser.parse_args()


shutil.rmtree(studypath(ARGS.name))
shutil.rmtree(f"{ROOT}/pointnet/models/hpo/{ARGS.name}/")
