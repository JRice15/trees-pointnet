import os, sys
from pathlib import PurePath
from os.path import dirname as dirn

# three steps up: pycrown, baselines, root
ROOT = PurePath(dirn(dirn(dirn(os.path.abspath(__file__)))))
# make 'shared' visible
sys.path.append(ROOT.as_posix())
