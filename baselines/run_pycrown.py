"""
PyCrown - Fast raster-based individual tree segmentation for LiDAR data
-----------------------------------------------------------------------
Copyright: 2018, Jan Zörner
Licence: GNU GPLv3
"""

from datetime import datetime

from pycrown import PyCrown

from shapely.geometry import mapping, Point, Polygon
import pandas as pd
import geopandas as gpd
import numpy as np

import fiona

def export_tree_locations(PC, loc='top'):
    """ Convert tree top raster indices to georeferenced 3D point shapefile
    Parameters
    ----------
    loc :     str, optional
              tree seed position: `top` or `top_cor`
    """
    outfile = PC.outpath / f'tree_location_{loc}.shp'
    outfile.parent.mkdir(parents=True, exist_ok=True)

    if outfile.exists():
        outfile.unlink()

    schema = {
        'geometry': '3D Point',
        'properties': {'DN': 'int', 'TH': 'float'}
    }
    with fiona.collection(
        str(outfile), 'w', 'ESRI Shapefile', schema, crs=PC.srs #crs_wkt=PC.srs
    ) as output:
        for tidx in range(len(PC.trees)):
            feat = {}
            tree = PC.trees.iloc[tidx]
            elevation = np.nan_to_num(float(tree[f'{loc}_elevation']))
            feat['geometry'] = mapping(
                Point(tree[loc].x, tree[loc].y, elevation)
            )
            TH = np.nan_to_num(float(tree[f'{loc}_height']))
            feat['properties'] = {'DN': tidx,
                                  'TH': TH}
            output.write(feat)

def filter_trees(PC, path, loc='top', exclude=False):
    points = [Point(l.x,l.y) for l,z in zip(PC.trees[loc],PC.trees[f'{loc}_elevation'])]
    gdf = gpd.read_file(path)
    good = [gdf.contains(point).any() for point in points]
    if exclude:
        good = [not g for g in good]
    PC.trees = PC.trees[good]

def read_trees(PC, path):
    trees = gpd.read_file(path)
    df = pd.DataFrame(np.array([trees.geometry, trees.geometry], dtype='object').T,
                          dtype='object', columns=['top_cor', 'top'])
    print(df)
    PC.trees = PC.trees.append(df)
    print(PC.trees)

if __name__ == '__main__':
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dsm',required=True)
    parser.add_argument('--dtm',required=True)
    parser.add_argument('--chm',required=True)
    parser.add_argument('--points')
    parser.add_argument('--trees',default='',help='csv file with trees to use as input')
    parser.add_argument('--area')
    parser.add_argument('--bldgs')
    parser.add_argument('--out', required=True)
    parser.add_argument('--min_dist', type=int)
    parser.add_argument('--threshold_rel', type=float)
    args = parser.parse_args()

    TSTART = datetime.now()

    F_CHM = args.chm
    F_DTM = args.dtm
    F_DSM = args.dsm
    F_LAS = args.points

    print('making PyCrown object')
    PC = PyCrown(F_CHM, F_DTM, F_DSM, F_LAS, outpath=args.out)

    # Cut off edges
    # PC.clip_data_to_bbox((1802200, 1802400, 5467250, 5467450))

    # Smooth CHM with 5m median filter
    print('running median filter')
    PC.filter_chm(3, ws_in_pixels=True)

    if args.trees is not '':
        print('reading trees from file')
        # read trees from file
        read_trees(PC, args.trees)
    else:
        # Tree Detection with local maximum filter
        print('running tree detection')
        PC.tree_detection(PC.chm, ws=5, ws_in_pixels=True, hmin=1.,
                          min_dist=args.min_dist,
                          threshold_rel=args.threshold_rel)

    # Clip trees to bounding box (no trees on image edge)
    # original extent: 1802140, 1802418, 5467295, 5467490
    # PC.clip_trees_to_bbox(bbox=(1802150, 1802408, 5467305, 5467480))
    # PC.clip_trees_to_bbox(bbox=(1802160, 1802400, 5467315, 5467470))
    print('running edge clipping')
    PC.clip_trees_to_bbox(inbuf=11)  # inward buffer of 11 metre
    
    # remove trees outside of area
    print('filtering out by area')
    if args.area is not None:
        filter_trees(PC,args.area,exclude=False)
    if args.bldgs is not None:
        filter_trees(PC,args.bldgs,exclude=True)

    export_tree_locations(PC,loc='top')

    # Crown Delineation
    print('running crown delineation')
    PC.crown_delineation(algorithm='dalponteCIRC_numba', th_tree=1.,
                         th_seed=0.7, th_crown=0.55, max_crown=10.)

    # Correct tree tops on steep terrain
    print('correcting tree tops')
    PC.correct_tree_tops()

    # Calculate tree height and elevation
    print('getting tree height and elevation')
    PC.get_tree_height_elevation(loc='top')
    PC.get_tree_height_elevation(loc='top_cor')

    export_tree_locations(PC,loc='top_cor')

    PC.trees.to_csv(PC.outpath / 'trees.csv')

    # Screen small trees
    #PC.screen_small_trees(hmin=20., loc='top')

    # Convert raster crowns to polygons
    # print('converting to raster crowns')
    # PC.crowns_to_polys_raster()
    # #print('converting to smooth raster crowns')
    # PC.crowns_to_polys_smooth(store_las=False)

    # # Check that all geometries are valid
    # print('quality control')
    # PC.quality_control()

    # print(PC.trees)

    # # Export results
    # print('export')
    # PC.export_raster(PC.chm, PC.outpath / 'chm.tif', 'CHM')
    # PC.export_tree_locations(loc='top')
    # PC.export_tree_locations(loc='top_cor')
    # PC.export_tree_crowns(crowntype='crown_poly_raster')
    # PC.export_tree_crowns(crowntype='crown_poly_smooth')

    TEND = datetime.now()

    print(f"Number of trees detected: {len(PC.trees)}")
    print(f'Processing time: {TEND-TSTART} [HH:MM:SS]')
