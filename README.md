# Urban Trees Pointnet

## Setting up

Create a conda environment with tensorflow 2.4 (or 2.4-gpu), using the provided
yml file


## Training Modes

* count: predict the count of trees in each input patch
* mmd: mean max discrepancy loss. Outputs an arbitrary set of points with weights representing how likely each point is to be a tree
* pwtt: pointwise treetop (deprecated)
* pwmmd: mmd, but predicted for every input point. Outputs an isATree weight for each input point


## How to run

### data preprocessing

`cd data_preprocessing`

`python3 filter_lidar.py --infile <input las file> --outfile <output filename>`  
With raw las/laz file, add height above ground (HAG) dimension, filter to HAG>0, and optionally downsample

`python3 chunked_lidar_to_patches`  
Split lidar to grid patches. Optional `--subdivide` argument, to subdivide grid into smaller squares.
This script outputs `data/train_patches.h5` and `data/test_patches.h5`

### training

from the root directory of this project:

`python3 train.py --mode <mode> --name <name>` and many other command-line options available


### evaluation

Evaluation is run automatically after training. You can also run it on its own, with:

`python3 evaluate.py --name <name>` with many command-line options available
