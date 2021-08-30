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
import rasterio
import tables


if not laspy.__version__.startswith("2"):
    print("This program requires laspy >=2.0. It can be installed like this:")
    print("$ pip3 install --pre laspy lazrs")


ORIG_ROWS = 44
ORIG_COLS = 49
# may differ from above because of subdivision of grid
ROWS = None
COLS = None
# length/width of a grid square
GRID_SIZE = None

def load_train_gt():
    """
    Load ground truth tree locations
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
    Load Test GT (test grid squares handpicked by Milo)
    """

    test_gt = gpd.read_file("../data/Patches/Patches.shp")
    test_gt.crs = "EPSG:4326"
    test_gt = test_gt.to_crs("EPSG:26911")

    test_gt_x = test_gt["geometry"].x
    test_gt_y = test_gt["geometry"].y

    test_gt = np.stack([test_gt_x, test_gt_y], axis=1)

    print("Test Ground Truth")
    print(test_gt)
    print("min:", test_gt.min(axis=0), "max:", test_gt.max(axis=0))

    return test_gt


def load_grid(subdivide):
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

    # subdividing
    grid_x = np.unique(np.linspace(grid_x[1:], grid_x[:-1], num=subdivide+1))
    grid_y = np.unique(np.linspace(grid_y[1:], grid_y[:-1], num=subdivide+1))

    # make sure the differences between all grid squares are the same
    global GRID_SIZE
    GRID_SIZE = np.diff(grid_x)[0]
    assert np.allclose(np.diff(grid_x), GRID_SIZE)
    assert np.allclose(np.diff(grid_y), GRID_SIZE)

    # make sure both are strictly increasing
    assert np.all(grid_x[1:] - grid_x[:-1] > 0)
    assert np.all(grid_y[1:] - grid_y[:-1] > 0)

    print("Grid square side length:", GRID_SIZE)
    print("Grid x, y")
    print(grid_x)
    print("min:", grid_x.min(axis=0), "max:", grid_x.max(axis=0))
    print(grid_y)
    print("min:", grid_y.min(axis=0), "max:", grid_y.max(axis=0))

    global ROWS, COLS
    COLS = len(grid_x)
    ROWS = len(grid_y)

    print(ROWS, "rows,", COLS, "cols")

    return grid_x, grid_y


def load_naip(naipfile, grid_x_y=None):
    with h5py.File("../data/test_patches.h5", "r") as f:
        COLS = f.attrs["gridcols"]
        ROWS = f.attrs["gridrows"]
        GRIDSIZE = f.attrs["gridsize"]
        MIN_X = f.attrs["grid_min_x"]
        MIN_Y = f.attrs["grid_min_y"]

    if grid_x_y is None:
        grid_x, grid_y = load_grid(int((ROWS - 1) / 44))
    else:
        grid_x, grid_y = grid_x_y

    # raster = rasterio.open("../../SaMo Trees Data/SaMo_NAIP_60cmm/SaMo_NAIP_60cm.tif")
    raster = rasterio.open(naipfile)
    NAIP_X_MIN, NAIP_Y_MIN, NAIP_X_MAX, NAIP_Y_MAX = raster.bounds
    assert str(raster.crs).lower() == "epsg:26911"
    im = raster.read()
    im = np.moveaxis(im, 0, -1) / 256 # channels last, float
    print("naip shape", im.shape)
    print("gridsize", GRIDSIZE)

    y,x,z = im.shape

    xpixelwidth = (NAIP_X_MAX - NAIP_X_MIN) / x
    ypixelwidth = (NAIP_Y_MAX - NAIP_Y_MIN) / y
    assert np.isclose(xpixelwidth, 0.6) and np.isclose(ypixelwidth, 0.6)
    pix_per_square = GRIDSIZE / round(xpixelwidth, 1)
    assert np.isclose(pix_per_square, 128)
    pix_per_square = int(pix_per_square)

    # create dict of patch ids to images
    out = {}
    for x in range(0, len(grid_x)):
        for y in range(0, len(grid_y)):
            y_ind, x_ind = raster.index(grid_x[x], grid_y[y])
            patch = im[y_ind:y_ind+pix_per_square, x_ind-pix_per_square:x_ind]
            out["patch{}_{}".format(x,y)] = patch

    return out, raster


def naip2ndvi(im):
    nir = im[...,3]
    red = im[...,0]
    ndvi = (nir - red) / (nir + red)
    return ndvi

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
    # searchsorted returns first index where value could be inserted for sorted order.
    # y==0, y==ROWS, x==0, or x=COLS will result in that point falling outside the grid.
    for y in range(1, ROWS):
        for x in range(1, COLS):
            these = pts[(xlocs == x) & (ylocs == y)]
            seperated[y][x] = these if len(these) else None
    return seperated


def make_patcher(fp, group, grid_x, grid_y, inds=None):
    """factory function to make add_to_patches func"""
    def add_to_patches(sep_pts, raster=None):
        """
        take the output from seperate_pts_by_grid (list of np.array|None), and add each to the corresponding patch
        args:
            sep_pts: nested list of points seperated by grid square (see seperate_pts_by_grid)
            raster: optional 4 channel naip raster
        """
        nonlocal fp, group, grid_x, grid_y, inds
        atom = tables.Float32Atom()
        if inds is None:
            inds = [(x,y) for x in range(COLS) for y in range(ROWS)]
        for (x,y) in inds:
            pt_group = sep_pts[y][x]
            if pt_group is not None:
                # # normalize x,y points to 0-1
                # pt_group[:,0] = (pt_group[:,0] - grid_x[x] + GRID_SIZE) / GRID_SIZE
                # pt_group[:,1] = (pt_group[:,1] - grid_y[y] + GRID_SIZE) / GRID_SIZE
                # normalize zs
                # if zmin is not None:
                #     pt_group[:,2] = (pt_group[:,2] - zmin) / (zmax - zmin)
                if raster is not None:
                    xys = pt_group[:,:2]
                    naip = raster.sample(xys) # returns generator
                    # convert to float
                    naip = np.array([i for i in naip]) / 256
                    ndvi = naip2ndvi(naip).reshape(-1, 1)
                    pt_group = np.concatenate([pt_group, ndvi], axis=-1)
                try:
                    earray = fp.get_node("/"+group+"/patch{}_{}".format(x,y))
                except tables.NoSuchNodeError:
                    earray = fp.create_earray("/"+group, "patch{}_{}".format(x,y), atom, (0, pt_group.shape[1]))
                earray.append(pt_group)
    return add_to_patches


transformer = pyproj.Transformer.from_crs("EPSG:3857", "EPSG:26911", 
                area_of_interest=pyproj.transformer.AreaOfInterest(-118.55, 33.97, -118.40, 34.08))
def reproject(xs, ys):
    xs, ys = transformer.transform(xs, ys)
    return xs, ys


def main():
    very_start_time = time.perf_counter()
    parser = argparse.ArgumentParser()
    parser.add_argument("--lasfile",required=True,help="input las file")
    parser.add_argument("--naipfile",required=True,help="input naip tif file")
    parser.add_argument("--subdivide",type=int,default=1,help="number of times to split each grid square")
    ARGS = parser.parse_args()

    """
    Use pytables to create extendable h5 file
    inspired by https://stackoverflow.com/questions/30376581/save-numpy-array-in-append-mode
    """

    gt = load_train_gt()
    test_gt = load_test_gt()
    grid_x, grid_y = load_grid(ARGS.subdivide)

    train_fp = tables.open_file("../data/train_patches.h5", "w")
    test_fp = tables.open_file("../data/test_patches.h5", "w")

    # add groups and attributes
    for fp in (train_fp, test_fp):
        fp.create_group("/", "gt")
        fp.create_group("/", "lidar")
        fp.create_group("/", "naip")

        fp.get_node("/")._v_attrs["lasfile"] = ARGS.lasfile
        fp.get_node("/")._v_attrs["naipfile"] = ARGS.naipfile
        fp.get_node("/")._v_attrs["gridrows"] = ROWS
        fp.get_node("/")._v_attrs["gridrows"] = ROWS
        fp.get_node("/")._v_attrs["gridrows"] = ROWS
        fp.get_node("/")._v_attrs["gridcols"] = COLS
        fp.get_node("/")._v_attrs["gridsize"] = GRID_SIZE
        fp.get_node("/")._v_attrs["grid_min_x"] = grid_x.min()
        fp.get_node("/")._v_attrs["grid_min_y"] = grid_y.min()


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

    print("test patches:", len(test_patch_inds))
    print("train patches:", len(train_patch_inds))

    # create patchers
    train_lidar_patcher = make_patcher(train_fp, "lidar", grid_x, grid_y, inds=train_patch_inds)
    train_gt_patcher = make_patcher(train_fp, "gt", grid_x, grid_y, inds=train_patch_inds)
    test_lidar_patcher = make_patcher(test_fp, "lidar", grid_x, grid_y, inds=test_patch_inds)
    test_gt_patcher = make_patcher(test_fp, "gt", grid_x, grid_y, inds=test_patch_inds)

    # get NAIP
    naip_patches, naip_raster = load_naip(ARGS.naipfile, grid_x_y=(grid_x, grid_y))

    # add gt to patches
    train_gt_patcher(sep_train_gt)
    test_gt_patcher(filtered_sep_test_gt)

    chunk_size = 1_000_000
    count = 0
    z_max = None
    with laspy.open(ARGS.lasfile, "r") as reader:
        while True:
            pts = reader.read_points(chunk_size)
            if pts is None:
                break
            count += len(pts)

            # Note to future: never use pts["X"], only pts["x"]. the capitalized version scales the numbers to 
            # remove the decimal bc that's how laspy stores the underlying data
            t1 = time.perf_counter()
            xs, ys = reproject(pts["x"], pts["y"])
            zs = pts["HeightAboveGround"]
            if z_max is None:
                z_max = np.percentile(zs, 99.5)
            else:
                z_max = max(np.percentile(zs, 99.5), z_max)
            pts = np.stack([xs, ys, zs], axis=1)
            # filter negative zs
            pts = pts[pts[...,2] >= -1]
            # filter too large zs
            pts = pts[pts[...,2] < 120]

            print("  reprojection:", time.perf_counter() - t1, "sec")

            t1 = time.perf_counter()
            sep_pts = seperate_pts_by_grid(pts, grid_x, grid_y)
            print("  seperation:", time.perf_counter() - t1, "sec")

            t1 = time.perf_counter()
            train_lidar_patcher(sep_pts, naip_raster)
            test_lidar_patcher(sep_pts, naip_raster)
            print("  add to patches:", time.perf_counter() - t1, "sec")

            print(count, "points complete")

            train_fp.flush()
            test_fp.flush()

    """
    add NAIP images
    """
    print("Adding NAIP patches")
    t1 = time.perf_counter()
    for patchname in train_fp.get_node("/lidar")._v_children.keys():
        train_fp.create_array("/naip", patchname, naip_patches[patchname])
    for patchname in test_fp.get_node("/lidar")._v_children.keys():
        test_fp.create_array("/naip", patchname, naip_patches[patchname])
    print("  done: ", time.perf_counter() - t1, "sec")
    
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
    train_fp.get_node("/lidar")._v_attrs["z_max"] = z_max

    print("min, max gt trees in train patches:", min(train_gt_lens), max(train_gt_lens))
    train_fp.get_node("/gt")._v_attrs["min_trees"] = min(train_gt_lens + test_gt_lens)
    train_fp.get_node("/gt")._v_attrs["max_trees"] = max(train_gt_lens + test_gt_lens)

    print("min, max points in test patches:", min(test_patch_lens), max(test_patch_lens))
    test_fp.get_node("/lidar")._v_attrs["min_points"] = min(test_patch_lens)
    test_fp.get_node("/lidar")._v_attrs["max_points"] = max(test_patch_lens)
    test_fp.get_node("/lidar")._v_attrs["z_max"] = z_max

    print("min, max gt trees in test patches:", min(test_gt_lens), max(test_gt_lens))
    test_fp.get_node("/gt")._v_attrs["min_trees"] = min(train_gt_lens + test_gt_lens)
    test_fp.get_node("/gt")._v_attrs["max_trees"] = max(train_gt_lens + test_gt_lens)

    train_fp.close()
    test_fp.close()

    print("Total time:", (time.perf_counter() - very_start_time)/60, "min")


if __name__ == "__main__":
    main()
