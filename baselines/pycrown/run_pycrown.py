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




def pycrown_predict_treetops(chm, dtm, dsm, outpath, params, area=None, bldgs=None):
    """
    args:
        params: dict with these keys:
            chm_smooth_ws: windows size of chm medium smoothing (pixels) 
            plm_ws: peak local max windowsize (pixels)
            plm_min_dist: min pixels between tree peaks
            plm_threshold_abs: meters, min height of trees
            inbuf_m: meters inward to disallow trees
            cdl_*: crown deliniation parameters:
                tree: 0-1
                seed: 0-1
                crown: 0-1
                maxcrown: max width of crown in meters
    """

    PC = PyCrown(chm_file=chm, dtm_file=dtm, dsm_file=dsm, las_file=None, outpath=outpath)

    # Smooth CHM with median filter
    if params["chm_smooth_pix"] > 0:
        PC.filter_chm(
            params["chm_smooth_pix"], 
            ws_in_pixels=True, 
            circular=True)

    # Tree Detection with local maximum filter
    PC.tree_detection(PC.chm, 
                        ws=params["plm_ws"],
                        ws_in_pixels=True,
                        # hmin=1.,
                        min_dist=params["plm_min_dist"],
                        threshold_abs=params["plm_threshold_abs"])

    # Clip trees to bounding box (no trees on image edge)
    if params["inbuf_m"] > 0:
        PC.clip_trees_to_bbox(inbuf=params["inbuf_m"]) # inward buffer of 11 metre
    
    # remove trees outside of area
    if area is not None:
        filter_trees(PC,area,exclude=False)
    if bldgs is not None:
        filter_trees(PC,bldgs,exclude=True)

    # Crown Delineation
    PC.crown_delineation(algorithm='dalponteCIRC_numba', 
                        th_tree=params["cdl_th_tree"], # 1.0
                        th_seed=params["cdl_th_seed"], # 0.7, 
                        th_crown=params["cdl_th_crown"], # 0.55, 
                        max_crown=params["cdl_th_maxcrown"], # 10.
                    )

    # Correct tree tops on steep terrain
    PC.correct_tree_tops()

    return PC




if __name__ == '__main__':
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dsm',required=True)
    parser.add_argument('--dtm',required=True)
    parser.add_argument('--chm',required=True)
    parser.add_argument('--points')
    # parser.add_argument('--trees',default='',help='csv file with trees to use as input')
    parser.add_argument('--area')
    parser.add_argument('--bldgs')
    parser.add_argument('--out', required=True)
    parser.add_argument('--min_dist', type=int)
    parser.add_argument('--threshold_abs', type=float)
    args = parser.parse_args()

    params = {
        "chm_smooth_pix": 3,
        "plm_ws": 5,
        "plm_min_dist": args.min_dist,
        "plm_threshold_abs": args.threshold_abs,
        "inbuf_m": 11,
        "cdl_th_tree": 1.0,
        "cdl_th_seed": 0.7,
        "cdl_th_crown": 0.55,
        "cdl_th_maxcrown": 10.,
    }

    PC = pycrown_predict_treetops(args.chm, args.dtm, args.dsm, args.out, params)

    export_tree_locations(PC, loc="top_cor")