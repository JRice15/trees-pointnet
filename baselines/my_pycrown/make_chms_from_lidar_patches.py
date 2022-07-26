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
from pathlib import PurePath
import tempfile

def run_pdal(pipeline):
    with tempfile.NamedTemporaryFile('w', delete=False) as f:
        json.dump({"pipeline": pipeline}, f)
    #print(f.name)
    subprocess.run(['pdal', 'pipeline', f.name])


def calc_chm(dsm_path,dtm_path,chm_path,block_size=1024):
    if os.path.exists(chm_path):
        os.remove(chm_path)
    with rasterio.open(dsm_path) as dsm:
        meta = dsm.meta
        nodata = dsm.nodata
        H,W = dsm.height,dsm.width
        with rasterio.open(dtm_path) as dtm:
            with rasterio.vrt.WarpedVRT(dtm,**meta) as dtm_vrt:
                assert dsm.height == dsm.width == dtm.height == dtm.width == 256, f"{dsm.height}, {dsm.width}, {dtm.height}, {dtm.width}"
                with rasterio.open(chm_path,'w',**meta) as chm:
                    for r in range(0,H,block_size):
                        for c in range(0,W,block_size):
                            dsm_block = dsm.read(1, window=Window(c, r, block_size, block_size))
                            dtm_block = dtm_vrt.read(1, window=Window(c, r, block_size, block_size))
                        
                            chm_block = dsm_block - dtm_block
                            chm_block[dsm_block==dsm.nodata] = nodata
                            chm_block[dtm_block==dtm.nodata] = nodata
                            
                            chm.write(chm_block,window=Window(c,r,chm_block.shape[1],chm_block.shape[0]),indexes=1)



def make_dsm(args, inputs, dsm_path, write_params):
    if os.path.exists(dsm_path):
        os.remove(dsm_path)

    dsm_pipeline = copy(inputs)
    dsm_pipeline.append(
        {
            "type":"filters.range",
            "limits":"returnnumber[1:1]"
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
            "window_size": 20,
            **write_params,
        })

    print('  running dsm pipeline...')
    run_pdal(dsm_pipeline)


def make_dtm(args, inputs, dtm_path, write_params):
    # dtm pipeline
    if os.path.exists(dtm_path):
        os.remove(dtm_path)
    
    dtm_pipeline = copy(inputs)
    dtm_pipeline.append(
        {
            "type":"filters.range",
            "limits":"Classification[2:2]"
        })

    dtm_pipeline.append(
        {
            "type": "writers.gdal",
            "filename":dtm_path,
            "output_type":"idw",
            "data_type":"float32",
            "gdaldriver":"GTiff",
            "window_size": 20,
            **write_params,
        })

    print('  running dtm pipeline...')
    run_pdal(dtm_pipeline)



parser = argparse.ArgumentParser(description='Produce DTM,DSM,CHM, and merged LAZ file from input LAS/LAZ files')
parser.add_argument('--lidar-dir', required=True, help='dir containing .las/.laz files')
parser.add_argument("--naip-dir",required=True)
parser.add_argument('--output-dir', required=True, help='output directory')
parser.add_argument('--resolution', type=float, default=0.6,
                    help='output resolution (m)') 
parser.add_argument('--delaunay', action='store_true',
                    help='use delaunay triangulation for DTM') 
args = parser.parse_args()

inputs = glob.glob(os.path.join(args.lidar_dir,"*.las")) + glob.glob(os.path.join(args.lidar_dir,"*.laz"))

assert len(inputs), "no las files found in dir"


naip_files = glob.glob(os.path.join(args.naip_dir, "*.tif"))

for naip in naip_files:
    num = PurePath(naip).stem.split("_")[-1]
    print("patch", num)
    with rasterio.open(naip) as raster:
        left = raster.bounds.left
        bottom = raster.bounds.bottom
        right = raster.bounds.right
        top = raster.bounds.top    
        bounds = ([left, right], [bottom, top])

        write_params = {
            #"bounds": str(bounds),
            "resolution": args.resolution,
            "width": raster.width,
            "height": raster.height,
            "origin_x": left,
            "origin_y": bottom,
        }

    inputs = [os.path.join(args.lidar_dir, f"lidar_{num}.las")]

    dsm_path = os.path.join(args.output_dir, f"dsm_{num}.tif")
    dtm_path = os.path.join(args.output_dir, f"dtm_{num}.tif")
    chm_path = os.path.join(args.output_dir, f"chm_{num}.tif")

    make_dsm(args, inputs, dsm_path=dsm_path, write_params=write_params)
    make_dtm(args, inputs, dtm_path=dtm_path, write_params=write_params)

    calc_chm(dsm_path, dtm_path, chm_path)

print('done.')
