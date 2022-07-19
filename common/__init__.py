"""
for basic procedures (data handling, consistent evaluation methods) that need to
be used by both the baselines and my model
"""
import os, sys
from os.path import dirname as dirn
from pathlib import PurePath

REPO_ROOT = PurePath(dirn(dirn(os.path.abspath(__file__))))
DATA_DIR = REPO_ROOT.joinpath("data")