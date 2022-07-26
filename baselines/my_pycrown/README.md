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

Get only the LIDAR that falls within a NAIP tile (~1 hr):
```
python3 lidar_to_patches.py --las <REGION LAS> --naip-dir <REGION NAIP DIR> --out-format REGION-all-patches.las
```

Split into individual tiles using no-stream mode (~1 min)
```
python3 lidar_to_patches.py --las REGION-all-patches.las --naip-dir <REGION NAIP DIR> --out-format lidar_patches/lidar_#.las
```
(The # is a placeholder than will be filled in with the patch number)

Make DST, DTM, and CHM tiles:
```
python3 make_chms_from_lidar_patches.py --lidar-dir lidar_patches/ --naip-dir <REGION NAIP DIR> --output-dir chms/
```

Adapt paths as nescessary, you might need to create target directories beforehand
