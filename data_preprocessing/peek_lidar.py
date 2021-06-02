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

from chunked_lidar_to_patches import load_grid

with h5py.File("../data/train_patches.h5", "r") as f:
    ROWS = f.attrs["gridrows"]
    COLS = f.attrs["gridcols"]

class ARGS:
    subdivide = int((ROWS - 1) / 44)

grid_x, grid_y = load_grid(ARGS)


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
plt.close()

with tables.open_file("../data/train_patches.h5", "r") as train_fp, \
        tables.open_file("../data/test_patches.h5", "r") as test_fp:

    train_attrs = train_fp.get_node("/lidar")._v_attrs
    test_attrs = test_fp.get_node("/lidar")._v_attrs
    for i in train_attrs._f_list():
        print("train", i, train_attrs[i])
    for i in test_attrs._f_list():
        print("test", i, test_attrs[i])
    train_attrs = train_fp.get_node("/gt")._v_attrs
    test_attrs = test_fp.get_node("/gt")._v_attrs
    for i in train_attrs._f_list():
        print("train", i, train_attrs[i])
    for i in test_attrs._f_list():
        print("test", i, test_attrs[i])

    train_patch_lens = [i.shape[0] for i in train_fp.get_node("/lidar")]
    train_gt_lens = [i.shape[0] for i in train_fp.get_node("/gt")]
    test_patch_lens = [i.shape[0] for i in test_fp.get_node("/lidar")]
    test_gt_lens = [i.shape[0] for i in test_fp.get_node("/gt")]

    pts_min = min(min(train_patch_lens), min(test_patch_lens))
    pts_max = max(max(train_patch_lens), max(test_patch_lens))
    step = (pts_max - pts_min) // 20
    plt.hist([train_patch_lens, test_patch_lens], label=["train", "test"], density=True, bins=range(pts_min, pts_max+1, step))
    plt.legend()
    plt.savefig("output/pts_per_patch")
    plt.close()

    trees_min = min(min(train_gt_lens), min(test_gt_lens))
    trees_max = max(max(train_gt_lens), max(test_gt_lens))
    step = (trees_max - trees_min) // 20
    plt.hist([train_gt_lens, test_gt_lens], label=["train", "test"], density=True, bins=range(trees_min, trees_max+1, step))
    plt.legend()
    plt.savefig("output/trees_per_patch")
    plt.close()

    train_fp.close()
    test_fp.close()



with h5py.File("../data/train_patches.h5", "r") as f:
    assert len(f["lidar"].keys()) == len(f["gt"].keys())


if ARGS.subdivide == 1:
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