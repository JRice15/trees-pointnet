import argparse
import glob
import os
import subprocess
import tempfile

dirn = os.path.dirn
ROOT = PurePath(dirn(dirn(os.path.abspath(__file__))))

parser = argparse.ArgumentParser()
parser.add_argument("--region",required=True)
parser.add_argument("--chm-dir",required=True)
parser.add_argument("--out-dir",required=True)
ARGS = parser.parse_args()

outdir = os.path.join(ARGS.out_dir)
os.makedirs(outdir, exist_ok=True)


for naip_path in glob.glob((ROOT / f"data/NAIP_patches/{region}/*.tif").as_posix()):
    patch_num = PurePath(naip_path).stem.split("_")[-1]

    bounds_file = tempfile.NamedTemporaryFile(suffix=".gpkg")
    subprocess.run(
        f"gdaltindex {bounds_file.name} {naip_path}"
    )

    for chm_path in glob.glob(os.path.join(ARGS.chm_path, "*.tif")):
        chm_name = PurePath(chm_path).stem
        out_file = os.path.join(ARGS.out_dir, chm_name + "_" + patch_num + ".tif")
        subprocess.run(
            f"gdalwarp -cutline {bounds_file.name} -crop_to_cutlin {chm_path} {out_file}"
        )

