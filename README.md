# Urban Trees Pointnet

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
