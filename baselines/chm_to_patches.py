import argparse
import glob
import os
import subprocess
import tempfile
from pathlib import PurePath

dirn = os.path.dirname
ROOT = PurePath(dirn(dirn(os.path.abspath(__file__))))

parser = argparse.ArgumentParser()
parser.add_argument("--region",required=True)
parser.add_argument("--chm-dir",required=True)
parser.add_argument("--out-dir",required=True)
ARGS = parser.parse_args()

outdir = os.path.join(ARGS.out_dir)
os.makedirs(outdir, exist_ok=True)

bounds_file = "_TEMP_CHM_TILE_BOUNDS_TEMP.gpkg"
if os.path.exists(bounds_file):
    os.remove(bounds_file)

for naip_path in glob.glob((ROOT / f"data/NAIP_patches/{ARGS.region}/*.tif").as_posix()):
    patch_num = PurePath(naip_path).stem.split("_")[-1]

    subprocess.run(
        f"gdaltindex {bounds_file} {naip_path}",
        shell=True,
    )

    for chm_path in glob.glob(os.path.join(ARGS.chm_dir, "*.tif")):
        chm_name = PurePath(chm_path).stem
        out_file = os.path.join(ARGS.out_dir, chm_name + "_" + patch_num + ".tif")
        subprocess.run(
            f"gdalwarp -cutline {bounds_file} -crop_to_cutline {chm_path} {out_file}",
            shell=True,
        )

    os.remove(bounds_file)
