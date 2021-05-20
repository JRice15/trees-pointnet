import numpy as np
import h5py
import re
import tables
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn

seaborn.set()
# import tensorflow as tf
# from dataset import make_data_generators

from chunked_lidar_to_patches import load_grid, seperate_pts_by_grid, add_to_patches, load_train_gt

load_train_gt()

grid_x, grid_y = load_grid()
from chunked_lidar_to_patches import ROWS, COLS


fig, axlist = plt.subplots(2,2)
for i,fname in enumerate(("train", "test")):
    for j,kind in enumerate(("lidar", "gt")):
        ax = axlist[i][j]
        plt.sca(ax)

        full_lens = []
        with tables.open_file("../data/{}_patches.h5".format(fname), "r") as train_fp:
            for y in range(ROWS):
                full_lens.append([])
                for x in range(COLS):
                    try:
                        num = train_fp.get_node("/{}/patch{}_{}".format(kind,x,y)).shape[0]
                    except:
                        num = 0
                    full_lens[-1].append(num)

        x, y = np.meshgrid(grid_y, grid_x, indexing='xy')
        plt.pcolormesh(x, y, full_lens, shading='auto')
        plt.colorbar()
        plt.title("{} {}".format(fname, kind))

plt.savefig("output/patches_raster.png", dpi=200)
plt.show()



with h5py.File("../data/train_patches.h5", "r") as f:
    assert len(f["lidar"].keys()) == len(f["gt"].keys())


"""
verify correct patches are selected for test
"""
with h5py.File("../data/test_patches.h5", "r") as f:
    keys = list(f["gt"].keys())
    keys = [re.sub(r"patch", "", i) for i in keys]
    keys = [i.split("_") for i in keys]
    keys = [(int(x), int(y)) for x,y in keys]
    keys.sort()

# a few of the test patches, hand indexed. These are indexed from 1, like the grid
expected = [
    (25,7),
    (29,8),
    (32,10),
    (21,10),
    (25,14)
]

expected.sort()
assert all([i in keys for i in expected])