import argparse
import json
import re
import shutil
import timeit
import os
from pprint import pprint

import geopandas as gpd
import pandas as pd
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pdal
import shapely

parser = argparse.ArgumentParser()
parser.add_argument("--bounds",help="shp file to crop points to")
args = parser.parse_args()

gt = gpd.read_file("data/SaMo_trees.csv")

geometries = "POINT (" + gt["Longitude"].astype(str) + " " + gt["Latitude"].astype(str) + ")"
geometries = geometries.apply(shapely.wkt.loads)
gt = gt.set_geometry(geometries)["geometry"]

gt.crs = "EPSG:4326"
gt = gt.to_crs("EPSG:3857")

x = gt.apply(lambda a: a.x)
y = gt.apply(lambda a: a.y)

gt = pd.DataFrame()
gt["X"] = x
gt["Y"] = y

gt = gt.to_numpy()

print("GT:")
print(gt)

OUTPUT_COLUMS = ["X", "Y", "HeightAboveGround"]


def read_with_bounds():
    shp = gpd.read_file(args.bounds).to_crs("EPSG:3857")
    wkts = [i.wkt for i in shp["geometry"]]

    # # NOTE this code below makes a WKT form polygon array, but PDAL segfaults using it for some reason
    # # turn list of wkt polygons into one string of a wkt polygon array
    # wkts = [re.sub(r"POLYGON \(\(", "(", i) for i in wkts]
    # wkts = [re.sub(r"\)\)", ")", i) for i in wkts]
    # coords = "POLYGON (" + ", ".join(wkts) + ")"


    # load the las, find the points inside each polygon

    filtered_points = None
    for i in range(len(shp)):
        # get polygons in wkt form
        coords = wkts[i]

        # print(coords)

        pipeline_json = [
            "data/SaMo_testregion4.las",
            {
                "type": "filters.hag_nn"
            },
            # {
            #     "type": "filters.crop",
            #     "polygon": coords
            # }
        ]

        pipeline = pdal.Pipeline(json.dumps(pipeline_json))
        pipeline.execute()

        assert len(pipeline.arrays) == 1

        arr = pipeline.arrays[0][OUTPUT_COLUMS]

        print(str(i) + ":", len(arr), "points")

        if filtered_points is None:
            filtered_points = arr
        else:
            filtered_points = np.concatenate([filtered_points, arr])

        # print(filtered_points)
        # print(filtered_points.dtype)
    return filtered_points


def read_without_bounds():
    pipeline_json = [
        "data/SaMo_testregion4.las",
        {
            "type": "filters.hag_nn"
        },
        # {
        #     "type": "filters.crop",
        #     "polygon": coords
        # }
    ]

    pipeline = pdal.Pipeline(json.dumps(pipeline_json))
    pipeline.execute()

    assert len(pipeline.arrays) == 1

    return pipeline.arrays[0][OUTPUT_COLUMS]


if args.bounds is None:
    print("no bounds provided, reading all points")
    points = read_without_bounds()
else:
    points = read_with_bounds()

x = points["X"]
y = points["Y"]
z = points["HeightAboveGround"]
points = np.vstack([x, y, z]).T.astype(np.float32)


print()
print(len(points), "total points")

print("  Min, Max, Max-Min")
cols = ["X", "Y", "Z"]
for i in range(3):
    print(cols[i], points[:,i].min(), points[:,i].max(), 
        points[:,i].max() - points[:,i].min())


xmin = points[:,0].min()
xmax = points[:,0].max()

ymin = points[:,1].min()
ymax = points[:,1].max()


print("Making patches")
# the units here are lat/lon degrees
PATCH_SIZE = 25
OVERLAP = 5

patches = []
patch_gts = []

y = ymin
while y < ymax:
    row = points[
        (points[:,1] >= y) & (points[:,1] < y + PATCH_SIZE)
    ]
    gt_row = gt[
        (gt[:,1] >= y) & (gt[:,1] < y + PATCH_SIZE)
    ]
    x = xmin
    while x < xmax:
        # print(x, y)
        patch = row[
                    (row[:,0] >= x) & (row[:,0] < x + PATCH_SIZE)
                ]
        patch_gt = gt_row[
                    (gt_row[:,0] >= x) & (gt_row[:,0] < x + PATCH_SIZE)
                ]
        if len(patch) > 20:
            patches.append(patch)
            patch_gts.append(patch_gt)
        x += PATCH_SIZE - OVERLAP
    y += PATCH_SIZE - OVERLAP

print(len(patches), "patches")
p_lens = [len(i) for i in patches]
print("min, max points in patches:", min(p_lens), max(p_lens))
gt_lens = [len(i) for i in patch_gts]
print("min, max gt trees in patches:", min(gt_lens), max(gt_lens))

plt.hist(p_lens)
plt.show()
plt.hist(gt_lens)
plt.show()


with h5py.File('data/patches.h5', 'w') as hf:
    grp_lidar = hf.create_group('lidar')
    grp_gt = hf.create_group('gt')
    for i,p in enumerate(patches):
        grp_lidar.create_dataset('patch'+str(i), data=p)
        grp_gt.create_dataset('patch'+str(i), data=patch_gts[i])
    grp_lidar.attrs["max_points"] = max(p_lens)
    grp_gt.attrs["max_trees"] = max(gt_lens)
