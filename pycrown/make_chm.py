"""
use pdal env
"""

from copy import copy
import argparse
import glob
import os
import subprocess
import json
import rasterio
from rasterio.windows import Window
import tempfile

def run_pdal(pipeline):
    with tempfile.NamedTemporaryFile('w', delete=False) as f:
        json.dump({"pipeline": pipeline}, f)
    print(f.name)
    subprocess.run(['pdal', 'pipeline', f.name])


def calc_chm(dsm_path,dtm_path,chm_path,block_size=1024):
    with rasterio.open(dsm_path) as dsm:
        meta = dsm.meta
        nodata = dsm.nodata
        H,W = dsm.height,dsm.width
        with rasterio.open(dtm_path) as dtm:
            with rasterio.vrt.WarpedVRT(dtm,**meta) as dtm_vrt:
                print(dsm.height,dsm.width,dtm.height,dtm.width)
                with rasterio.open(chm_path,'w',**meta) as chm:
                    for r in range(0,H,block_size):
                        for c in range(0,W,block_size):
                            dsm_block = dsm.read(1, window=Window(c, r, block_size, block_size))
                            dtm_block = dtm_vrt.read(1, window=Window(c, r, block_size, block_size))
                        
                            chm_block = dsm_block - dtm_block
                            chm_block[dsm_block==dsm.nodata] = nodata
                            chm_block[dtm_block==dtm.nodata] = nodata
                            
                            chm.write(chm_block,window=Window(c,r,chm_block.shape[1],chm_block.shape[0]),indexes=1)

parser = argparse.ArgumentParser(description='Produce DTM,DSM,CHM, and merged LAZ file from input LAS/LAZ files')
parser.add_argument('--input', required=True, help='input file or directory of input files')
parser.add_argument('--input_srs', default='EPSG:26911', help='input SRS')
parser.add_argument('--output', required=True, help='output directory')
parser.add_argument('--output_srs', default='EPSG:26911', help='output SRS') 
parser.add_argument('--resolution', type=float, default=0.6,
                    help='output resolution (m)') 
parser.add_argument('--delaunay', action='store_true',
                    help='use delaunay triangulation for DTM') 
args = parser.parse_args()

if os.path.isdir(args.input):
    inputs = glob.glob(os.path.join(args.input,"*.las")) + glob.glob(os.path.join(args.input,"*.laz"))
else:
    inputs = [args.input]

# dsm pipeline
dsm_path = os.path.join(args.output,"dsm.tif")
if not os.path.exists(dsm_path):
    dsm_pipeline = copy(inputs)
    dsm_pipeline.append(
        {
            "type":"filters.range",
            "limits":"returnnumber[1:1]"
        })
    dsm_pipeline.append(
        {
            "type":"filters.reprojection",
            "in_srs":args.input_srs,
            "out_srs":args.output_srs
        })
    #dsm_pipeline.append(
        #{
            #"type":"filters.outlier",
            #"method":"statistical",
            #"mean_k":12,
            #"multiplier":2.2
        #})
    dsm_pipeline.append(
        {
            "type":"filters.outlier",
            "method":"radius",
            "radius":1.0,
            "min_k":4
        })
    dsm_pipeline.append(
        {
            "type":"filters.range",
            "limits":"Classification![7:7]" # remove points classified as outliers
        })
    dsm_pipeline.append(
        {
            "type": "writers.gdal",
            "filename":dsm_path,
            "output_type":"idw",
            "data_type":"float32",
            "gdaldriver":"GTiff",
            "resolution": args.resolution,
            "window_size": 20
        })

    print('running dsm pipeline...')
    run_pdal(dsm_pipeline)

# dtm pipeline
dtm_path = os.path.join(args.output,"dtm.tif")
if not os.path.exists(dtm_path):
    dtm_pipeline = copy(inputs)
    dtm_pipeline.append(
        {
            "type":"filters.range",
            "limits":"Classification[2:2]"
        })
    dtm_pipeline.append(
        {
            "type":"filters.reprojection",
            "in_srs":args.input_srs,
            "out_srs":args.output_srs
        })
    if args.delaunay:
        dtm_pipeline.append(
        {
            "type": "filters.delaunay"
        })
        dtm_pipeline.append(
            {
                "type": "filters.faceraster",
                "resolution": args.resolution
            })
        dtm_pipeline.append(
            {
                "type": "writers.raster",
                "filename": dtm_path,
                "data_type":"float32"
            })
    else:
        # get bounds of DSM
        with rasterio.open(dsm_path) as dsm:
            left = dsm.bounds.left
            bottom = dsm.bounds.bottom
            right = dsm.bounds.right
            top = dsm.bounds.top
            H,W = dsm.height,dsm.width
            origin_x = dsm.transform.c
            origin_y = dsm.transform.f

        dtm_pipeline.append(
            {
                "type": "writers.gdal",
                "filename":dtm_path,
                "output_type":"idw",
                "data_type":"float32",
                "gdaldriver":"GTiff",
                #"bounds":f"([{left},{right}],[{bottom},{top}])",
                #"width":W,
                #"height":H,
                #"origin_x":origin_x,
                #"origin_y":origin_y,
                "resolution": args.resolution,
                "window_size": 20
            })

    print('running dtm pipeline...')
    run_pdal(dtm_pipeline)

# laz pipeline
# laz_path = os.path.join(args.output,"points.laz")
# laz_pipeline = copy(inputs)
# laz_pipeline.append(
#     {
#         "type":"filters.reprojection",
#         "in_srs":args.input_srs,
#         "out_srs":args.output_srs
#     })
# laz_pipeline.append(
#     {
#         "type": "writers.las",
#         "filename":laz_path,
#         "compression":"lazperf"
#     })

# if laz_path != args.input:
#     print('running laz pipeline...')
#     run_pdal(laz_pipeline)

print('computing chm...')
chm_path = os.path.join(args.output,'chm.tif')
calc_chm(dsm_path,dtm_path,chm_path)

print('done.')