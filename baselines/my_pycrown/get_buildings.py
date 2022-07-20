import osmnx as ox
import rasterio as rio
from rasterio import features
import argparse
import numpy as np
import os

parser = argparse.ArgumentParser()
parser.add_argument('--area',default='Santa Monica, CA',help='name of area')
parser.add_argument('--epsg',type=int,default=26911,help='EPSG code')
parser.add_argument('--out',help='path to output directory')
args = parser.parse_args()

# area = ox.geocode_to_gdf(args.area)
# area = area.to_crs(epsg=args.epsg)
# area.to_file(os.path.join(args.out,'area.gpkg'),driver='GPKG')

bldgs = ox.geometries_from_place(args.area,{'building':True})
bldgs = bldgs.to_crs(epsg=args.epsg)
bldgs = bldgs.applymap(lambda x: str(x) if isinstance(x, list) else x)
bldgs.drop(labels='nodes', axis=1).to_file(os.path.join(args.out,'bldgs.gpkg'), driver='GPKG')
