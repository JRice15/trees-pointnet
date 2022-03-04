import json
import os
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

dirn = os.path.dirname
ROOT = dirn(dirn(os.path.abspath(__file__)))


class KilledTrial(Exception): pass

class ignore_kbint():
    """
    ignore keyboard interrupts
    """

    def __enter__(self):
        self.old_handler = signal.signal(signal.SIGINT, self.handler)
                
    def handler(self, sig, frame):
        logging.debug('SIGINT received. Ignoring.')
    
    def __exit__(self, type, value, traceback):
        signal.signal(signal.SIGINT, self.old_handler)


def get_study(name, assume_exists):
    """
    get optuna study. If actual existence is different from assume_exists, error is raised
    args:
        name: str
        assume_exists: bool
    """
    os.makedirs(f"{ROOT}/hpo/studies", exist_ok=True)
    storage = optuna.storages.RDBStorage(
        url=f"sqlite:///{ROOT}/hpo/studies/{name}.db",
        engine_kwargs={"connect_args": {"timeout": 15}}, # longer than usual timeout is ok, because 15s is insignificant compared to the time each trial takes
    )
    if assume_exists:
        try:
            return optuna.load_study(
                study_name=name,
                storage=storage,
            )
        except Exception as e:
            raise ValueError(f"Error loading study '{name}': {str(e)}")
    else:
        try:
            return optuna.create_study(
                study_name=name,
                storage=storage,
                direction="maximize",
            )
        except optuna.exceptions.DuplicatedStudyError:
            raise ValueError(f"Study named '{name}' already exists. Supply the --resume flag if you want to resume it")



