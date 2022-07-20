#!/bin/bash

# create base environment
conda env create -n pycrown -f environment.yml
# add optuna (only availabke via conda-forge)
conda install --yes -n pycrown -c conda-forge optuna alembic==1.0.0