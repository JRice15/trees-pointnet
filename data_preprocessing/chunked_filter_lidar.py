import numpy as np 
import laspy
import tables
import time
import shapely
import h5py
import geopandas as gpd
import pyproj

if not laspy.__version__.startswith("2"):
    print("This program requires laspy >=2.0. It can be installed like this:")
    print("$ pip3 install --pre laspy lazrs")


ROWS = 44
COLS = 49


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



"""
Load Grid
"""

grid = gpd.read_file("../data/grid.gpkg")
grid.crs = "EPSG:26911"

# the grid is perfectly aligned with the CRS, so the polygon bounds give us the 
# edges of each grid square
grid_bounds = grid["geometry"].bounds.to_numpy()
# find larger valued x and y side of each grid square
grid_x = np.maximum(grid_bounds[:,2], grid_bounds[:,0])
grid_y = np.maximum(grid_bounds[:,3], grid_bounds[:,1])

grid_x = grid_x[:COLS] # first row; all row x's are the same
grid_y = grid_y[::COLS] # first column; all col y's are the same
grid_y = grid_y[::-1] # put y in sorted order

print("Grid x, y")
print(grid_x)
print("min:", grid_x.min(axis=0), "max:", grid_x.max(axis=0))
print(grid_y)
print("min:", grid_y.min(axis=0), "max:", grid_y.max(axis=0))


# make sure the differences between all grid square centers are the same
assert np.allclose(np.diff(grid_x), 153.6)
assert np.allclose(np.diff(grid_y), 153.6)

# make sure both are strictly increasing
assert np.all(grid_x[1:] - grid_x[:-1] > 0)
assert np.all(grid_y[1:] - grid_y[:-1] > 0)


def seperate_pts_by_grid(pts):
    """
    seperates an array of N points, shape (N,2+), where 2+ corresponds to x,y followed
    by any other dimensions, into each grid square
    returns:
        list, the same length as the number of grid squares
    """
    seperated = [None] * (ROWS * COLS)
    xlocs = np.searchsorted(grid_x, pts[:,0])
    ylocs = np.searchsorted(grid_y, pts[:,1])
    indexes = (ylocs * COLS) + xlocs
    print("i", indexes)
    for i in range(ROWS * COLS):
        these = pts[indexes == i]
        seperated[i] = these if len(these) else None
    return seperated




"""
Use pytables to create extendable h5 file
inspired by https://stackoverflow.com/questions/30376581/save-numpy-array-in-append-mode
"""

train_fp = tables.open_file("train_patches.h5", "w")
test_fp = tables.open_file("test_patches.h5", "w")
atom = tables.Float32Atom()
# create train data
train_fp.create_group("/", "gt")
train_fp.create_group("/", "lidar")
test_fp.create_group("/", "gt")
test_fp.create_group("/", "lidar")


def add_to_patches(fp, group, sep_pts, inds=None):
    """
    take the output from seperate_pts_by_grid (list of np.array|None), and add each to the corresponding patch
    """
    if inds is None:
        inds = range(len(sep_pts))
    for i in inds:
        pt_group = sep_pts[i]
        if pt_group is not None:
            try:
                earray = fp.get_node("/"+group+"/patch"+str(i))
            except tables.NoSuchNodeError:
                earray = fp.create_earray("/"+group, "patch"+str(i), atom, (0, pt_group.shape[1]))
            earray.append(pt_group)


# add test gt
sep_test_gt = seperate_pts_by_grid(test_gt)
# exclude points that fall very slightly outside their intended patch
sep_test_gt = [i if (i is None or len(i) > 3) else None for i in sep_test_gt]
add_to_patches(test_fp, "gt", sep_test_gt)

test_patch_inds = [i for i,v in enumerate(sep_test_gt) if v is not None]
train_patch_inds = [i for i,v in enumerate(sep_test_gt) if v is None]
print("test patches:", len(test_patch_inds))
print("train patches:", len(train_patch_inds))

# add train gt
sep_train_gt = seperate_pts_by_grid(gt)
add_to_patches(train_fp, "gt", sep_train_gt, inds=train_patch_inds)


transformer = pyproj.Transformer.from_crs("EPSG:3857", "EPSG:26911", 
                area_of_interest=pyproj.transformer.AreaOfInterest(-118.55, 33.97, -118.40, 34.08))
def reproject(xs, ys):
    xs, ys = transformer.transform(xs, ys)
    return xs, ys

chunk_size = 10_000_000
count = 0
with laspy.open("../data/SaMo_full_hag_subsampled_30x.laz", "r") as reader:
    while True:
        pts = reader.read_points(chunk_size)
        if pts is None:
            break
        count += len(pts)

        # Note to future: never use pts["X"], only pts["x"]. the capitalized version scales the numbers to remove the decimal for some reason
        xs, ys = reproject(pts["x"], pts["y"])
        pts = np.stack([xs, ys, pts["HeightAboveGround"]], axis=1)

        sep_pts = seperate_pts_by_grid(pts)
        # print(sep_pts)

        add_to_patches(train_fp, "lidar", sep_pts, inds=train_patch_inds)
        add_to_patches(test_fp, "lidar", sep_pts, inds=test_patch_inds)

        print(count, "points complete")


train_fp.close()
test_fp.close()

