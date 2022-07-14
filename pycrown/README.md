# Pycrown baseline

All code adapted from https://github.com/jonathanventura/lidar-processing/

# Environments:

to create Pycrown env:

```
conda create --name pycrown -c conda-forge python==3.6
conda run -n pycrown pip install -r requirements.txt

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