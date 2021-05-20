import argparse
import os
import sys
import time

import geopandas as gpd
import h5py
import laspy
import numpy as np
import pyproj
import shapely
import tables

parser = argparse.ArgumentParser()
parser.add_argument("--no-mpl",action="store_true")
parser.add_argument("--subdivide",type=int,default=1)
args = parser.parse_args()

if not laspy.__version__.startswith("2"):
    print("This program requires laspy >=2.0. It can be installed like this:")
    print("$ pip3 install --pre laspy lazrs")


ORIG_ROWS = 44
ORIG_COLS = 49
ROWS = None
COLS = None

def load_train_gt():
    """
    Load GT
    """

    gt = gpd.read_file("../data/SaMo_trees.csv")

    # some lat/lon locations are (0,0). Remove those
    gt = gt.loc[gt["Longitude"].astype(float) != 0]

    geometries = "POINT (" + gt["Longitude"].astype(str) + " " + gt["Latitude"].astype(str) + ")"
    geometries = geometries.apply(shapely.wkt.loads)
    gt = gt.set_geometry(geometries)["geometry"]

    gt.crs = "EPSG:4326"
    gt = gt.to_crs("EPSG:26911")

    x = gt.apply(lambda a: a.x).to_numpy()
    y = gt.apply(lambda a: a.y).to_numpy()

    gt = np.stack([x, y], axis=1)

    print("All Ground Truth:")
    print(gt)

    print("min:", gt.min(axis=0), "max:", gt.max(axis=0))

    return gt


def load_test_gt():
    """
    Load Test GT (Grid Squares Handpicked by Milo)
    """

    test_gt = gpd.read_file("../data/Annotated.shp")
    test_gt.crs = "EPSG:4326"
    test_gt = test_gt.to_crs("EPSG:26911")

    test_gt_x = test_gt["geometry"].x
    test_gt_y = test_gt["geometry"].y

    test_gt = np.stack([test_gt_x, test_gt_y], axis=1)

    print("Test Ground Truth")
    print(test_gt)
    print("min:", test_gt.min(axis=0), "max:", test_gt.max(axis=0))

    return test_gt


def load_grid():
    """ 
    Load Grid
    """
    grid = gpd.read_file("../data/grid.gpkg")
    grid.crs = "EPSG:26911"

    # the grid is perfectly aligned with the CRS, so the polygon bounds give us the 
    # edges of each grid square
    grid_bounds = grid["geometry"].bounds.to_numpy()
    # find larger valued x and y side of each grid square
    grid_x = np.maximum(grid_bounds[:,0], grid_bounds[:,2])
    grid_y = np.maximum(grid_bounds[:,1], grid_bounds[:,3])

    grid_lowerbound_x = np.minimum(grid_bounds[:,0], grid_bounds[:,2]).min()+0.6
    grid_lowerbound_y = np.minimum(grid_bounds[:,1], grid_bounds[:,3]).min()+0.6

    grid_x = grid_x[:ORIG_COLS] # first row; all row x's are the same
    grid_y = grid_y[::ORIG_COLS] # first column; all col y's are the same
    grid_y = grid_y[::-1] # put y in sorted order

    grid_x = np.insert(grid_x, 0, grid_lowerbound_x)
    grid_y = np.insert(grid_y, 0, grid_lowerbound_y)

    # make sure the differences between all grid squares are the same
    assert np.allclose(np.diff(grid_x), 153.6)
    assert np.allclose(np.diff(grid_y), 153.6)

    # make sure both are strictly increasing
    assert np.all(grid_x[1:] - grid_x[:-1] > 0)
    assert np.all(grid_y[1:] - grid_y[:-1] > 0)

    # subdividing
    grid_x = np.unique(np.linspace(grid_x[1:], grid_x[:-1], num=args.subdivide+1))
    grid_y = np.unique(np.linspace(grid_y[1:], grid_y[:-1], num=args.subdivide+1))

    print("Grid x, y")
    print(grid_x)
    print("min:", grid_x.min(axis=0), "max:", grid_x.max(axis=0))
    print(grid_y)
    print("min:", grid_y.min(axis=0), "max:", grid_y.max(axis=0))

    global ROWS, COLS
    ROWS = len(grid_x)
    COLS = len(grid_y)

    print(ROWS, "rows,", COLS, "cols")

    return grid_x, grid_y


def seperate_pts_by_grid(pts, grid_x, grid_y):
    """
    seperates an array of N points, shape (N,2+), where 2+ corresponds to x,y followed
    by any other dimensions, into each grid square
    returns:
        list, the same length as the number of grid squares
    """
    seperated = [[None] * COLS for i in range(ROWS)]
    xlocs = np.searchsorted(grid_x, pts[:,0])
    ylocs = np.searchsorted(grid_y, pts[:,1])
    # pts outside grid are excluded. searchsorted returns first index where value could be inserted for sorted order
    # y==0, y==ROWS, x==0, x=COLS means that point falls outside the grid.
    for y in range(1, ROWS):
        for x in range(1, COLS):
            these = pts[(xlocs == x) & (ylocs == y)]
            seperated[y][x] = these if len(these) else None
    return seperated


def add_to_patches(fp, group, sep_pts, inds=None):
    """
    take the output from seperate_pts_by_grid (list of np.array|None), and add each to the corresponding patch
    """
    atom = tables.Float32Atom()
    if inds is None:
        inds = [(x,y) for x in range(COLS) for y in range(ROWS)]
    for (x,y) in inds:
        pt_group = sep_pts[y][x]
        if pt_group is not None:
            try:
                earray = fp.get_node("/"+group+"/patch{}_{}".format(x,y))
            except tables.NoSuchNodeError:
                earray = fp.create_earray("/"+group, "patch{}_{}".format(x,y), atom, (0, pt_group.shape[1]))
            earray.append(pt_group)


transformer = pyproj.Transformer.from_crs("EPSG:3857", "EPSG:26911", 
                area_of_interest=pyproj.transformer.AreaOfInterest(-118.55, 33.97, -118.40, 34.08))
def reproject(xs, ys):
    xs, ys = transformer.transform(xs, ys)
    return xs, ys


def main():
    """
    Use pytables to create extendable h5 file
    inspired by https://stackoverflow.com/questions/30376581/save-numpy-array-in-append-mode
    """

    gt = load_train_gt()
    test_gt = load_test_gt()
    grid_x, grid_y = load_grid()

    train_fp = tables.open_file("../data/train_patches.h5", "w")
    test_fp = tables.open_file("../data/test_patches.h5", "w")
    atom = tables.Float32Atom()
    # create train data
    train_fp.create_group("/", "gt")
    train_fp.create_group("/", "lidar")
    test_fp.create_group("/", "gt")
    test_fp.create_group("/", "lidar")


    # seperate test gt
    sep_test_gt = seperate_pts_by_grid(test_gt, grid_x, grid_y)
    sep_train_gt = seperate_pts_by_grid(gt, grid_x, grid_y)

    # find indexes of test and train patches
    # exclude test patches that only contain a few points, ie points that fall very slightly outside their intended patch
    filtered_sep_test_gt = []
    test_patch_inds = []
    train_patch_inds = []
    for y in range(ROWS):
        filtered_sep_test_gt.append([])
        for x in range(COLS):
            test_v = sep_test_gt[y][x]
            train_v = sep_train_gt[y][x]
            if test_v is None or len(test_v) <= 3:
                filtered_sep_test_gt[-1].append(None)
                if (train_v is not None) and (len(train_v) > 3):
                    train_patch_inds.append((x,y))
            else:
                filtered_sep_test_gt[-1].append(test_v)
                test_patch_inds.append((x,y))
    add_to_patches(test_fp, "gt", filtered_sep_test_gt)

    print("test patches:", len(test_patch_inds))
    print("train patches:", len(train_patch_inds))

    # add train gt
    add_to_patches(train_fp, "gt", sep_train_gt, inds=train_patch_inds)

    chunk_size = 10_000_000
    count = 0
    with laspy.open("../data/SaMo_full_hag_subsampled_30x.laz", "r") as reader:
        while True:
            pts = reader.read_points(chunk_size)
            if pts is None:
                break
            count += len(pts)

            # Note to future: never use pts["X"], only pts["x"]. the capitalized version scales the numbers to 
            # remove the decimal bc that's how laspy stores the underlying data
            t1 = time.time()
            xs, ys = reproject(pts["x"], pts["y"])
            pts = np.stack([xs, ys, pts["HeightAboveGround"]], axis=1)
            print("  reprojection:", time.time() - t1, "sec")

            t1 = time.time()
            sep_pts = seperate_pts_by_grid(pts, grid_x, grid_y)
            print("  seperation:", time.time() - t1, "sec")

            t1 = time.time()
            add_to_patches(train_fp, "lidar", sep_pts, inds=train_patch_inds)
            add_to_patches(test_fp, "lidar", sep_pts, inds=test_patch_inds)
            print("  add to patches:", time.time() - t1, "sec")

            print(count, "points complete")

            train_fp.flush()
            test_fp.flush()

    
    """
    save length attributes
    """

    train_patch_lens = [i.shape[0] for i in train_fp.get_node("/lidar")]
    train_gt_lens = [i.shape[0] for i in train_fp.get_node("/gt")]
    test_patch_lens = [i.shape[0] for i in test_fp.get_node("/lidar")]
    test_gt_lens = [i.shape[0] for i in test_fp.get_node("/gt")]

    print("min, max points in train patches:", min(train_patch_lens), max(train_patch_lens))
    train_fp.get_node("/lidar")._v_attrs["min_points"] = min(train_patch_lens)
    train_fp.get_node("/lidar")._v_attrs["max_points"] = max(train_patch_lens)

    print("min, max gt trees in train patches:", min(train_gt_lens), max(train_gt_lens))
    train_fp.get_node("/gt")._v_attrs["min_trees"] = min(train_gt_lens)
    train_fp.get_node("/gt")._v_attrs["max_trees"] = max(train_gt_lens)

    print("min, max points in test patches:", min(test_patch_lens), max(test_patch_lens))
    test_fp.get_node("/lidar")._v_attrs["min_points"] = min(test_patch_lens)
    test_fp.get_node("/lidar")._v_attrs["max_points"] = max(test_patch_lens)

    print("min, max gt trees in test patches:", min(test_gt_lens), max(test_gt_lens))
    test_fp.get_node("/gt")._v_attrs["min_trees"] = min(test_gt_lens)
    test_fp.get_node("/gt")._v_attrs["max_trees"] = max(test_gt_lens)

    if not args.no_mpl:
        import matplotlib.pyplot as plt
        os.makedirs("output", exist_ok=True)

        plt.hist(train_patch_lens)
        plt.savefig("output/train_pts_per_patch")
        plt.close()
        plt.hist(train_gt_lens)
        plt.savefig("output/train_trees_per_patch")
        plt.close()

        plt.hist(test_patch_lens)
        plt.savefig("output/test_pts_per_patch")
        plt.close()
        plt.hist(test_gt_lens)
        plt.savefig("output/test_trees_per_patch")
        plt.close()

    train_fp.close()
    test_fp.close()


if __name__ == "__main__":
    main()
