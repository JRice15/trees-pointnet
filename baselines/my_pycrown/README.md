# Pycrown baseline

The `pycrown` subfolder is all from [Pycrown](https://github.com/manaakiwhenua/pycrown), except for some edits I made: reduced printing output and one bug fix (pycrown.py:662).

Most other code adapted from: https://github.com/jonathanventura/lidar-processing/


# Environment Setup

to create Pycrown env:

```
./make_pycrown_env.sh

wget https://lastools.github.io/download/LAStools.zip
unzip LAStools.zip
cd LAStools
make
cp bin/laszip /data/$USER/miniconda/condabin/laszip
ln -s /data/$USER/miniconda/condabin/laszip /data/$USER/miniconda/condabin/laszip-cli
```

pdal:

```
conda create --name pdal -c conda-forge pdal rasterio pip
conda run -n pdal pip install osmnx>=1.0.0
```

# Data setup

1. Run `make_chm.py` for the Las files for each region to generate CHM, DSM, and DTM
2. Run `chm_to_patches.py` on the outputs to create raster tiles corresponding to our annotated NAIP tiles
3. Run `estimate_pycrown_params.py --train --specs <FILENAME> --ntrials <N>` to run N hyperoptimization trials to estimate the best pycrown parameters on the training dataset. Specs should be a file like `elcap_chm_specs.json`, which lists directories containing the outputs frm the previous step.
4. Run `estimate_pycrown_params.py --test --specs <FILENAME>` to evaluate the best found parameters on the test set.
