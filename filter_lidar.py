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
parser.add_argument("--bounds",required=True,help="shp file to crop points to")
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

print(gt)

OUTPUT_COLUMS = ["X", "Y", "HeightAboveGround"]

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
        {
            "type": "filters.crop",
            "polygon": coords
        }
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

print()
print(len(filtered_points), "total points")

print("  Min, Max, Max-Min")
for i in OUTPUT_COLUMS:
    print(i, filtered_points[i].min(), filtered_points[i].max(), 
        filtered_points[i].max() - filtered_points[i].min())


xmin = filtered_points["X"].min()
xmax = filtered_points["X"].max()

ymin = filtered_points["Y"].min()
ymax = filtered_points["Y"].max()


print("Making patches")
# the units here are lat/lon degrees
PATCH_SIZE = 10
OVERLAP = 1

patches = []
patch_gts = []

y = ymin
while y < ymax:
    x = xmin
    while x < xmax:
        # print(x, y)
        patch = filtered_points[
                    (filtered_points["X"] >= x) &
                    (filtered_points["X"] < x + PATCH_SIZE) &
                    (filtered_points["Y"] >= y) &
                    (filtered_points["Y"] < y + PATCH_SIZE)
                ]
        patch_gt = gt[
                    (gt["X"] >= x) &
                    (gt["X"] < x + PATCH_SIZE) &
                    (gt["Y"] >= y) &
                    (gt["Y"] < y + PATCH_SIZE)
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

# plt.hist(lens)
# plt.show()


with h5py.File('data/patches.h5', 'w') as hf:
    grp_lidar = hf.create_group('lidar')
    grp_gt = hf.create_group('gt')
    for i,p in enumerate(patches):
        grp_lidar.create_dataset('patch'+str(i), data=p)
        grp_gt.create_dataset('patch'+str(i), data=patch_gts[i])
    grp_lidar.attrs["max_points"] = max(p_lens)
    grp_gt.attrs["max_trees"] = max(gt_lens)
