import pandas as pd
import geopandas as gpd
import argparse


TO_CRS = 'EPSG:26911'


def read_test_trees(fname):
    trees = read_trees(fname)
    trees['point'] = trees['geometry']
    add_buffer(trees)
    return trees

def read_trees(fname):
    try:
        trees = read_csv(fname)
    except pd.errors.ParserError: # not a csv, must be shp/gpkg
        trees = read_gpkg(fname)
    trees = trees.to_crs(TO_CRS)
    trees = trees[['geometry']]
    return trees

def read_csv(fname):
    trees = pd.read_csv(fname)
    trees = gpd.GeoDataFrame(
        trees,
        geometry=gpd.points_from_xy(trees['Longitude'],
                                    trees['Latitude']),
        crs=REF_CRS)
    trees = trees[['geometry']]
    return trees

def read_pycrown(top_fname, crown_fname):
    test_tops = gpd.read_file(top_fname)
    test_tops['point'] = test_tops['geometry']
    test_crowns = gpd.read_file(crown_fname)
    test_crowns['buffer'] = test_crowns['geometry']
    assert test_tops.crs == test_crowns.crs

    test_trees = gpd.sjoin(test_tops, test_crowns, how='left', predicate='within')
    test_trees = test_trees[~test_trees.index.duplicated(keep='first')]
    test_trees['height'] = test_trees['TH']
    test_trees['buffer'] = test_trees.apply(
        lambda x: x['buffer'] if not pd.isnull(x['buffer']) else _calc_buffer(x),
        axis=1)
    test_trees = test_trees[['height', 'point', 'buffer', 'geometry']]
    return test_trees

def read_gpkg(fname):
    trees = gpd.read_file(fname)
    trees['point'] = trees['geometry']
    return trees

def add_buffer(trees):
    trees['buffer'] = trees.apply(_calc_buffer, axis=1)

def _calc_buffer(tree):
    return tree['point'].buffer(5)

def candidate_search(test_trees, ref_trees):
    """ 
    Parameters
    ----------
    test_trees: geopandas.GeoDataFrame
        Required columns: 'height' in meter (float),
                          'point' (shapely.geometry.point.Point),
                          'buffer' (shapely.geometry.polygon.Polygon)
    ref_trees: geopandas.GeoDataFrame
        Required columns: 'height' in meter (float),
                          'geometry' (shapely.geometry.point.Point)
    """
    assert test_trees.crs == ref_trees.crs
    test_trees['geometry'] = test_trees['buffer']
    result = gpd.sjoin(test_trees, ref_trees, how='inner', predicate='contains',
                       lsuffix='test', rsuffix='ref')
    result = result.reset_index().rename(columns={'index':'index_test'})\
        [['index_test', 'index_ref']]
    return result

def candidate_vote(test_trees, ref_trees, search_result):
    test_trees['geometry'] = test_trees['point']
    vote_result = search_result.groupby('index_test').apply(
        lambda x: _select_best_candidate(test_trees.loc[x.name],
                                         ref_trees.loc[x['index_ref']]))
    vote_result = vote_result.rename('index_ref').reset_index()
    return vote_result

def candidate_validate(test_trees, ref_trees, vote_result):
    validate_result = vote_result.groupby('index_ref').apply(
        lambda x: _select_best_candidate(ref_trees.loc[x.name],
                                        test_trees.loc[x['index_test']]))
    validate_result = validate_result.rename('index_test').reset_index()
    return validate_result

def _select_best_candidate(test_tree, ref_df):
    try:
        ref_df['distance'] = ref_df.distance(test_tree['geometry'])
    except ValueError:
        print(len(set(ref_df.index)), len(ref_df.index))
        print(ref_df)
        print(test_tree)
        raise
    nearest_ref = ref_df['distance'].idxmin()
    return nearest_ref

def metrics(test_trees, ref_trees, result):
    n_test = len(test_trees)
    n_ref = len(ref_trees)
    n_match = len(result)
    precision = n_match / n_test
    recall = n_match / n_ref
    f1 = 2 * precision * recall / (precision + recall)
    
    test_trees['geometry'] = test_trees['point']
    distances = test_trees.loc[result['index_test']].reset_index().\
        distance(ref_trees.loc[result['index_ref']].reset_index())
    rmse = (distances ** 2).mean() ** .5
    
    return {
        "precision": precision, 
        "recall": recall, 
        "f1": f1, 
        "rmse": rmse,
    }

def filter_by_aoi(aoi_gdf, trees):
    return gpd.sjoin(trees, aoi_gdf, how='inner', predicate='within')[trees.columns]

def main(gt_file, pred_file):
    test_trees = read_test_trees(pred_file)
    ref_trees = read_trees(gt_file)

    test_trees = test_trees[~test_trees.index.duplicated(keep='first')]
    ref_trees = ref_trees[~ref_trees.index.duplicated(keep='first')]

    assert len(test_trees) != 0 and len(ref_trees) != 0
    search_result = candidate_search(test_trees, ref_trees)
    assert len(search_result) != 0
    vote_result = candidate_vote(test_trees, ref_trees, search_result)
    result = candidate_validate(test_trees, ref_trees, vote_result)
    
    return metrics(test_trees, ref_trees, result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ref', type=str, required=True,
                        help='reference csv')
    parser.add_argument('--test_top', type=str,
                        help='test top csv/shp/gpkg')
    # parser.add_argument('--test_crown', type=str,
    #                     help='test crown shp/gpkg')
    # parser.add_argument('--aoi',
    #     help='shp/gpkg containing area of interest (tested area)')
    # parser.add_argument('--aoi_indices', type=int, nargs='+', default=None,
    #     help='indices of polygons in aoi, if not provided all polygons are used')
    args = parser.parse_args()

    from pprint import pprint
    pprint(main(args.ref, args.test_top))