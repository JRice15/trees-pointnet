# Urban Trees Pointnet

## Setting Up

Create a conda environment with tensorflow 2.4 (or 2.4-gpu), using the provided
yml file


## How to Run

### Data Preprocessing

`cd data_preprocessing`

With an environment that has `pdal` and `numpy`, run the following:

`python3 filter_lidar.py --infile <region las file> [--reproject ? --scale ?]` 
With raw las/laz file, compute height above ground (HAG) dimension, filter to HAG>0, and possibly reproject (note the --scale argument when reprojecting).

Create and activate a `laspy` environment with the provided .yml file, then:

`python3 generate_lidar_patches.py --specs ../data/elcap_lidar_specs.json --outname <dataset name>`
This selects all LIDAR points lying inside an annotated grid square, and saves it to a .npy file.

Optionally:
`python3 analyze_dataset.py --dsname <dataset name>` to generate some nice diagnostic plots and such, inside `data/lidar/<dataset name>`


### Training

First time only: `./docker_build.sh`

Then, use `./docker_run N` where N is the GPU index you would like to use to activate the docker container. If this is the first time you have ever activated the container, run `./compile_ops` inside as well.

Finally, run

`python3 train.py -h` to see the full array of command line options available for training


### Evaluation

Evaluation is run automatically after training. You can also run it on its own, with:

`python3 evaluate.py --name <name>` which evaluates the model with the given name
