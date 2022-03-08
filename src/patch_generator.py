import json
import os
import glob
import sys
import time
from pathlib import PurePath

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model
from tensorflow.keras import backend as K
from tensorflow.keras import layers

from src import ARGS, REPO_ROOT, DATA_DIR, LIDAR_CHANNELS
from src.utils import raster_plot, rotate_pts, get_all_regions, get_naipfile_path, Bounds

VAL_SPLIT = 0.1
TEST_SPLIT = 0.15

def subdivide_bounds(bounds_dict, n_subdivide):
    subdiv_bounds = {}
    if n_subdivide == 1:
        for (region, patch_num), bounds in bounds_dict.items():
            subdiv_bounds[(region, patch_num, 0)] = bounds
    else:
        for (region, patch_num), bounds in bounds_dict.items():
            left, right, bottom, top = bounds.xy_fmt()
            x_width = (right - left) / n_subdivide
            y_width = (top - bottom) / n_subdivide
            i = 0
            for x in np.linspace(left, right, n_subdivide+1)[:-1]:
                for y in np.linspace(bottom, top, n_subdivide+1)[:-1]:
                    new_bounds = Bounds.from_minmax([x, y, x+x_width, y+y_width])
                    subdiv_bounds[(region, patch_num, i)] = new_bounds
                    i += 1
    return subdiv_bounds


def filter_pts(bounds_dict, points_dict, keyfunc):
    """
    filter points into bounds
    args:
        bounds_dict: maps composite patch keys to patch bounds
        points_dict: maps possibly composite patch keys to np arrays of points
        keyfunc: translates the keys of bounds_dict to the keys of points_dict
    """
    out = {}
    for key, bound in bounds_dict.items():
        left,bott,right,top = bound.minmax_fmt()
        pts_key = keyfunc(key)
        pts = points_dict[pts_key]
        x = pts[:,0]
        y = pts[:,1]
        filtered = pts[
                (x >= left) & (x < right) & (y <= top) & (y > bott) 
            ]
        out[key] = filtered
    return out


class LidarPatchGen(keras.utils.Sequence):
    """
    keras Sequence that loads the dataset in batches
    """

    def __init__(self, patch_ids, name=None, batchsize=None, training=False):
        """
        args:
            patch_ids: list(tuple), where each tuple is of the format (region, patch number)
            name: examples: "train", "test", "val"
            batchsize: overrides ARGS.batchsize if given
            training: bool, whether this gen is being used for training (will augment data)
        in the ARGS global object:
            mode
            batchsize: int
            subdivide: int
            npoints: points to input per patch
            ndvi: bool, whether to include ndvi channel
            ragged: bool
        """
        self.name = name
        self.dsname = ARGS.dsname
        self.batch_size = ARGS.batchsize if batchsize is None else batchsize
        self.n_subdivide = ARGS.subdivide
        self.orig_patch_ids = patch_ids
        self.regions = list(set([i[0] for i in self.orig_patch_ids]))
        self.y_counts_only = False
        if ARGS.loss == "count":
            self.y_counts_only = True
        self.nattributes = len(LIDAR_CHANNELS)
        self.npoints = ARGS.npoints
        self.z_max = 50 # for normalization
        self.batch_time = 0
        self.training = training

        # initialize the data
        self.init_data()

    def init_random(self):
        """initialize or reinitialize the random number generatore"""
        self.random = np.random.default_rng(seed=999)

    def init_data(self):
        LIDAR_DIR = DATA_DIR.joinpath("lidar", ARGS.dsname, "regions")
        NAIP_DIR = DATA_DIR.joinpath("NAIP_patches")
        GT_DIR = DATA_DIR.joinpath("ground_truth_csvs")

        x_column_options = ("long_nad83", "long_utm11", "point_x", "x")
        y_column_options = ("lat_nad83", "lat_utm11", "point_y", "y")

        # load ground-truth trees
        orig_gt_trees = {}
        for region in self.regions:
            files = glob.glob(GT_DIR.joinpath(region + "*.csv").as_posix())
            assert len(files) == 1
            table = pd.read_csv(files[0])
            # lowercase the columns
            table.columns = [x.lower() for x in table.columns]
            x_col = None
            y_col = None
            for col in x_column_options:
                if col in table:
                    x_col = table[col].to_numpy()
                    break
            for col in y_column_options:
                if col in table:
                    y_col = table[col].to_numpy()
                    break
            if x_col is None or y_col is None:
                raise ValueError("Could not find correct gt columns for {}".format(region))
            gt_pts = np.stack([x_col, y_col], axis=-1)
            orig_gt_trees[region] = gt_pts

        # load bounds and lidar
        orig_bounds = {}
        orig_lidar = {}
        for (region,patch_num) in self.orig_patch_ids:
            naipfile = get_naipfile_path(region, patch_num)
            with rasterio.open(naipfile) as raster:
                orig_bounds[(region,patch_num)] = Bounds.from_minmax(raster.bounds)
            lidarfile = LIDAR_DIR.joinpath(region, "lidar_patch_{}.npy".format(patch_num)).as_posix()
            orig_lidar[(region,patch_num)] = np.load(lidarfile)

        # subdivide bounds
        subdiv_bounds = subdivide_bounds(orig_bounds, self.n_subdivide)
        
        # divide the lidar
        subdiv_lidar = filter_pts(subdiv_bounds, orig_lidar, keyfunc=lambda key: (key[0], key[1]) ) # keyfunc selects region and patchnum
        
        # free up space
        del orig_lidar

        # filter patches by number of lidar points
        orig_ids = list(subdiv_lidar.keys())
        self.patch_ids = [i for i in orig_ids if len(subdiv_lidar[i]) >= self.npoints]
        self.patch_bounds = {k:v for k,v in subdiv_bounds.items() if k in self.patch_ids}
        self.patched_lidar = {k:v for k,v in subdiv_lidar.items() if k in self.patch_ids}

        self.num_ids = len(self.patch_ids)
        self.num_filtered_ids = len(orig_ids) - self.num_ids
        if ARGS.test:
            self.num_ids = 2

        # filter gt trees into patches
        self.patched_gt = filter_pts(self.patch_bounds, orig_gt_trees, keyfunc=lambda x: x[0]) # keyfunc selects region

        # get max gt trees
        maxtrees = 0
        for pts in self.patched_gt.values():
            maxtrees = max(maxtrees, len(pts))
        self.max_trees = maxtrees

        # get normalization data
        self.norm_mins = {}
        self.norm_maxs = {}
        for patch_key, bound in self.patch_bounds.items():
            left,bott,right,top = bound.minmax_fmt()
            # spatial channels, then spectral channels. Last channel is NDVI
            min_xyz = [left, bott, 0] + [0, 0, 0, 0, -1]
            max_xyz = [right, top, self.z_max] + [1, 1, 1, 1, 1]
            self.norm_mins[patch_key] = np.array(min_xyz)
            self.norm_maxs[patch_key] = np.array(max_xyz)

        # normalize lidar and gt data
        for patch_id,pts in self.patched_lidar.items():
            # handle no-data values in spectral data
            spectral = pts[3:]
            spectral[spectral < -1e20] = -1
            pts[3:] = spectral
            # normalize lidar
            mins = self.norm_mins[patch_id]
            maxs = self.norm_maxs[patch_id]
            pts = (pts - mins) / (maxs - mins)
            self.patched_lidar[patch_id] = pts
            # normalize gt xy locs
            y_pts = self.patched_gt[patch_id]
            y_pts = (y_pts - mins[:2]) / (maxs[:2] - mins[:2])
            self.patched_gt[patch_id] = y_pts

        # sort for reproducibility
        self.sorted()
        # set random seed
        self.init_random()
        # deterministic random shuffle
        self.random.shuffle(self.patch_ids)

        # create batch objects
        self.x_batch_shape = (self.batch_size, self.npoints, self.nattributes)
        if self.y_counts_only:
            self.y_batch_shape = (self.batch_size,)
        else:
            self.y_batch_shape = (self.batch_size, self.max_trees, 3) # 3 -> (x,y,isvalid)

    def __len__(self):
        return self.num_ids // self.batch_size

    def __getitem__(self, idx, return_ids=False, no_rotate=False):
        """
        load one batch
        """
        t1 = time.perf_counter()
        idx = idx * self.batch_size
        end_idx = idx + self.batch_size

        X_batch = np.empty(self.x_batch_shape, dtype=K.floatx())
        Y_batch = np.zeros(self.y_batch_shape, dtype=K.floatx())

        for i, patch_key in enumerate(self.patch_ids[idx:end_idx]):
            # get pts
            lidar_patch = self.patched_lidar[patch_key]
            # select <npoints> evenly spaced points randomly from batch.
            #   this is done because indexing with an arbitrary array is 
            #   orders of magnitude slower than a simple slice
            num_x_pts = lidar_patch.shape[0]
            step = num_x_pts // self.npoints
            leftover = num_x_pts % self.npoints
            if leftover == 0:
                rand_offset = 0
            else:
                rand_offset = self.random.integers(leftover) # randomly generated int
            top_offset = leftover - rand_offset
            X_batch[i] = lidar_patch[rand_offset:num_x_pts-top_offset:step]
            
            # select all gt y points, or just y count
            y_pts = self.patched_gt[patch_key]
            n_y_pts = y_pts.shape[0]
            if self.y_counts_only:
                Y_batch[i] = n_y_pts
            else:
                min_xy, max_xy = min_xyz[:2], max_xyz[:2]
                Y_batch[i,:n_y_pts,2] = 1
                Y_batch[i,:n_y_pts,:2] = y_pts

        # shuffle input points within each patch
        #  this shuffles the points within each seperate patch in the same way, but that is random enough for me
        self.random.shuffle(X_batch, axis=1)

        if self.training:
            if not no_rotate:
                # augment by a random rotation
                rot_degrees = self.random.choice([0, 90, 180, 270])
                X_batch = rotate_pts(X_batch, degrees=rot_degrees)
                if not self.y_counts_only:
                    Y_batch = rotate_pts(Y_batch, degrees=rot_degrees)
            # random gaussian noise
            if ARGS.noise_sigma is not None:
                X_batch += self.random.normal(
                                        loc=0, 
                                        scale=ARGS.noise_sigma, 
                                        size=self.x_batch_shape)

        x = tf.constant(X_batch, dtype=K.floatx())
        y = tf.constant(Y_batch, dtype=K.floatx())

        self.batch_time += time.perf_counter() - t1
        if return_ids:
            return x, y, self.patch_ids[idx:end_idx]
        return x, y

    def load_all(self):
        """
        load all examples into one big batch. should be used for the validation
        patch generator, because keras doesn't accept a Sequence for val data
        """
        self.sorted()
        x_batches = []
        y_batches = []
        for i in range(len(self)):
            x, y = self[i]
            x_batches.append(x)
            y_batches.append(y)
        x = tf.concat(x_batches, axis=0)
        y = tf.concat(y_batches, axis=0)
        return x, y

    def get_patch(self, region, patch_num, subdiv_num):
        """
        get the full i'th patch of the entire sorted dataset, or from a specific region
        args:
            region: str, region name
            patch_num: int
        returns:
            x, y, patch_id
        """
        # set temporary sorted ids
        old_training = self.training
        old_ids = self.patch_ids
        old_batchsize = self.batch_size
        self.training = False
        self.sorted()
        self.batch_size = 1
        # find batch index
        key = (region, patch_num, subdiv_num)
        index = self.patch_ids.index(key)
        # load batch
        x, y = self[index]
        # restore correct values
        self.patch_ids = old_ids
        self.batch_size = old_batchsize
        self.training = old_training
        return x[0], y[0], key

    def get_naip(self, patch_id):
        """
        get naip image. Channels: R-G-B-NIR
        args:
            patch_id: tuple: (region, patch_num, subdiv_num)
        """
        bounds = self.patch_bounds[patch_id]
        # load naip file
        region, patch_num, subdiv_num = patch_id
        naipfile = get_naipfile_path(region, patch_num)
        with rasterio.open(naipfile) as raster:
            # https://gis.stackexchange.com/questions/336874/get-a-window-from-a-raster-in-rasterio-using-coordinates-instead-of-row-column-o
            im = raster.read(window=rasterio.windows.from_bounds(*bounds.minmax_fmt(), raster.transform))
        # channels last format
        im = np.moveaxis(im, 0, -1) / 255.0
        # sometimes it selects a row of pixels outside of the image, which results in spurious very large negative numbers
        im = np.clip(im, 0, 1)
        return im

    def get_patch_bounds(self, patch_id):
        """
        args:
            patch_id: tuple: (region, patch_num, subdiv_num)
        """
        return self.patch_bounds[patch_id]

    def get_batch_shape(self):
        """
        get x,y shape of one batch
        x: (batchsize; npoints or None if ragged; nattributes per point)
        """
        print("Batch shape x:", self.x_batch_shape, "y:", self.y_batch_shape)
        return self.x_batch_shape, self.y_batch_shape

    def denormalize_pts(self, pts, patch_id):
        """
        reverse the normalization of a set of points; ie, it goes from 0-1 scaled 
        back to georeferenced coordinates.
        """
        min_xyz = self.norm_mins[patch_id]
        max_xyz = self.norm_maxs[patch_id]
        n_channels = pts.shape[-1]
        min_xyz = min_xyz[:n_channels]
        max_xyz = max_xyz[:n_channels]
        return (pts * (max_xyz - min_xyz)) + min_xyz

    def summary(self):
        print("LidarPatchGen '{}' from dataset '{}' regions:".format(self.name, self.dsname),  ", ".join(self.regions))
        print(" ", self.num_ids, "patches, in", len(self), "batches, batchsize", self.batch_size)
        print(" ", self.npoints, "points per patch.", self.num_filtered_ids, "patches dropped for having too few lidar points")
        try:
            print("  xbatch shape:", self.x_batch_shape)
            print("  ybatch shape:", self.y_batch_shape)
        except AttributeError:
            pass

    def on_epoch_end(self):
        print("avg batch time:", self.batch_time / len(self))
        print("total batch time:", self.batch_time)
        self.batch_time = 0
        self.random.shuffle(self.patch_ids)

    def sorted(self):
        """put ids in a reproducable order (sorted order)"""
        self.patch_ids.sort()




def get_tvt_split(dsname, regions):
    """
    returns patch ids for the train, val, and test datasets
    it selects the same patches every time given the same split, by selecting 
    every Nth patch
    """
    val_step = int(1/VAL_SPLIT)
    test_step = int(1/TEST_SPLIT)

    train = []
    val = []
    test = []
    for region in regions:
        naipfiles = DATA_DIR.joinpath("NAIP_patches", region, "*.tif").as_posix()
        patch_nums = []
        for filename in glob.glob(naipfiles):
            patch_num = int(PurePath(filename).stem.split("_")[-1])
            patch_nums.append(patch_num)
        patches = [(region, x) for x in patch_nums]
        test += patches[::test_step]
        rest_patches = [x for x in patches if x not in test]
        val += rest_patches[::val_step]
        train += [x for x in rest_patches if x not in val]
    return train, val, test



def get_train_val_gens(dsname, regions, val_batchsize=None, train_batchsize=None):
    """
    returns:
        train Keras Sequence, val Sequence or raw data, test Sequence
    """
    train, val, test = get_tvt_split(dsname, regions)

    train_gen = LidarPatchGen(train, name="train", training=True, batchsize=train_batchsize)
    val_gen = LidarPatchGen(val, name="validation", batchsize=val_batchsize)
    return train_gen, val_gen

def get_test_gen(dsname, regions):
    train, val, test = get_tvt_split(dsname, regions)

    test_gen = LidarPatchGen(test, name="test", batchsize=1)
    return test_gen



if __name__ == "__main__":
    dsname = "fullchannels"
    regions = get_all_regions(dsname)
    train_gen, val_gen = get_train_val_gens(dsname, regions)
    test_gen = get_test_gen(dsname, regions)
    train_gen.summary()
    val_gen.summary()
    test_gen.summary()


