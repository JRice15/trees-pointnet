import glob
import json
import os
import sys
import time
from pathlib import PurePath

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K

from src import ARGS, DATA_DIR, LIDAR_CHANNELS
from src.utils import rotate_pts, scale_meters_to_0_1

from common.data_handling import (Bounds, get_all_regions, get_naip_bounds,
                                  load_gt_trees, load_naip, get_data_splits)
from common.utils import MyTimer

# max height for pts in meters
Z_MAX_CLIP = 50
# negative height to remove points below
Z_MIN_CLIP = -10

# each patch overlaps halfway with the neighbor to its right
OVERLAP = 2

def subdivide_bounds(bounds_dict, n_subdivide):
    """
    given a dict mapping full-size patch ids to bounds, make sliding-window
    sub-patches with side length 1/n_subdivide of the original
    """
    subdiv_bounds = {}
    for (region, patch_num), bounds in bounds_dict.items():
        left, right, bottom, top = bounds.xy_fmt()
        x_width = (right - left) / n_subdivide
        y_width = (top - bottom) / n_subdivide
        # if left=0, right=100, subdivide=2, and overlap=2, we want: 
        # left_edges=[0, 25, 50]
        # this corresponds to the two subdivided patches, from 0 to 50 and 50 to 100, and the
        # overlap patch between them, 25 to 75
        left_edges = np.linspace(left, right, (n_subdivide*OVERLAP)+1)[:-OVERLAP]
        bottom_edges = np.linspace(bottom, top, (n_subdivide*OVERLAP)+1)[:-OVERLAP]
        for i,y in enumerate(bottom_edges):
            for j,x in enumerate(left_edges):
                new_bounds = Bounds.from_minmax([x, y, x+x_width, y+y_width])
                subdiv_bounds[(region, patch_num, j, i)] = new_bounds
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
    for key, bounds in bounds_dict.items():
        pts_key = keyfunc(key)
        pts = points_dict[pts_key]
        out[key] = bounds.filter_pts(pts)
    return out


class LidarPatchGen(keras.utils.Sequence):
    """
    keras Sequence that loads the dataset in batches

    important data attributes:
        valid_patch_ids: ids of subdivided patches that are valid as model inputs, ie have not been dropped for having too few lidar points
        bounds_full/subdiv: dict mapping patch id to Bounds for each full or subdivided patch
        gt_full/subdiv: dict mapping patch id to ground truth tree locations in that patch
        lidar_subdiv: dict mapping patch it to lidar points in that patch (no 'full' version, as it is a waste of memory)
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
        if ARGS.test:
            patch_ids = patch_ids[:10]
        self.orig_patch_ids = patch_ids
        self.regions = list(set([i[0] for i in self.orig_patch_ids]))
        self.y_counts_only = False
        self.nattributes = 3 + len(ARGS.channels) # xyz + spectral
        self.npoints = ARGS.npoints
        self.batch_time = 0
        self.training = training

        # initialize the data
        print("preparing dataset", self.name, "...")
        #   get lidar channel mask
        self.channel_mask = [True, True, True] + [c in ARGS.channels for c in LIDAR_CHANNELS[3:]]
        print(" ", np.array(LIDAR_CHANNELS)[self.channel_mask])

        self.init_data()

        # sort for reproducibility
        self.sorted()
        # set random seed
        self.init_random()
        # deterministic random shuffle
        self.random.shuffle(self.valid_patch_ids)


    def init_random(self):
        """initialize or reinitialize the random number generatore"""
        self.random = np.random.default_rng(seed=999)

    def init_data(self):
        # set random seed
        self.init_random()

        LIDAR_DIR = DATA_DIR.joinpath("lidar", ARGS.dsname, "regions")
        NAIP_DIR = DATA_DIR.joinpath("NAIP_patches")

        # load ground-truth trees
        self.gt_full = load_gt_trees(self.orig_patch_ids)

        # load bounds and lidar

        self.bounds_full = {}
        lidar_full = {}
        for (region,patch_num) in self.orig_patch_ids:
            self.bounds_full[(region,patch_num)] = get_naip_bounds(region, patch_num)
            lidarfile = LIDAR_DIR.joinpath(region, "lidar_patch_{}.npy".format(patch_num)).as_posix()
            pts = np.load(lidarfile)
            # get selected channels
            if not all(self.channel_mask):
                pts = pts[:,self.channel_mask]
            # clip spurious Z values
            z = pts[:,2]
            pts = pts[(z > Z_MIN_CLIP) & (z <= Z_MAX_CLIP)]
            lidar_full[(region,patch_num)] = pts

        # subdivide bounds
        self.bounds_subdiv = subdivide_bounds(self.bounds_full, self.n_subdivide)
        # subdivide lidar
        self.lidar_subdiv = filter_pts(self.bounds_subdiv, lidar_full, keyfunc=lambda key: (key[0], key[1]) ) # keyfunc selects region and patchnum
        # subdivide gt trees
        self.gt_subdiv = filter_pts(self.bounds_subdiv, self.gt_full, lambda key: (key[0], key[1]) ) # keyfunc selects region and patchnum
        # free up space
        del lidar_full

        self.max_trees = max(len(x) for x in self.gt_subdiv.values())        

        # handle small patches
        # counters
        self.num_small_patches = 0
        self.num_pts_filled = 0
        self.max_pts_filled = 0
        self.smallest_patch = None
        # final patch data
        self.valid_patch_ids = []
        for patch_id,pts in self.lidar_subdiv.items():
            # look for too-small patches
            if len(pts) < self.npoints:
                # handle according to mode
                self.num_small_patches += 1
                needed = self.npoints - len(pts)
                # just skip this patch
                if ARGS.handle_small == "drop":
                    continue
                # fill with the rest with -1000 pts
                elif ARGS.handle_small == "fill":
                    filler_pts = np.full((needed, self.nattributes), -1000)
                    pts = np.concatenate([pts, filler_pts], axis=0)
                # duplicate points
                elif ARGS.handle_small == "repeat":
                    if len(pts) == 0:
                        # no points to duplicate, just drop this patch
                        continue
                    selected = self.random.choice(pts, needed, axis=0)
                    pts = np.concatenate([pts, selected], axis=0)
                else:
                    raise ValueError("Unknown handle small '{}'".format(ARGS.handle_small))

                # update lidar at this patch
                self.lidar_subdiv[patch_id] = pts

                self.num_pts_filled += needed
                if needed > self.max_pts_filled:
                    self.max_pts_filled = needed
                    self.smallest_patch = patch_id
            
            # if we get here, it means we didn't drop, so patch is valid
            self.valid_patch_ids.append(patch_id)
    
        self.num_ids = len(self.valid_patch_ids)

        # normalize lidar
        for patch_id,pts in self.lidar_subdiv.items():
            # handle no-data values in spectral data
            spectral = pts[:,3:]
            spectral[spectral < -1] = -1.0
            pts[:,3:] = spectral
            # normalize
            mins, maxs = self.get_norm_data(patch_id)
            pts = (pts - mins) / (maxs - mins)
            self.lidar_subdiv[patch_id] = pts
        # normalize gt
        for patch_id,y_pts in self.gt_subdiv.items():
            mins, maxs = self.get_norm_data(patch_id)
            y_pts = (y_pts - mins[:2]) / (maxs[:2] - mins[:2])
            self.gt_subdiv[patch_id] = y_pts

        # create batch objects
        self.x_batch_shape = (self.batch_size, self.npoints, self.nattributes)
        if self.y_counts_only:
            self.y_batch_shape = (self.batch_size,)
        else:
            self.y_batch_shape = (self.batch_size, self.max_trees, 3) # 3 -> (x,y,isvalid)

    def __len__(self):
        return self.num_ids // self.batch_size

    def __getitem__(self, idx, return_ids=False, no_augment=False):
        """
        load one batch
        """
        t1 = time.perf_counter()
        idx = idx * self.batch_size
        end_idx = idx + self.batch_size

        X_batch = np.empty(self.x_batch_shape, dtype=K.floatx())
        Y_batch = np.zeros(self.y_batch_shape, dtype=K.floatx())

        for i, patch_key in enumerate(self.valid_patch_ids[idx:end_idx]):
            # get pts
            lidar_patch = self.lidar_subdiv[patch_key]
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
            y_pts = self.gt_subdiv[patch_key]
            n_y_pts = y_pts.shape[0]
            if self.y_counts_only:
                Y_batch[i] = n_y_pts
            else:
                Y_batch[i,:n_y_pts,2] = 1
                Y_batch[i,:n_y_pts,:2] = y_pts

        # shuffle input points within each patch
        #  this shuffles the points within each seperate patch in the same way, but that is random enough for me
        self.random.shuffle(X_batch, axis=1)

        if self.training and (not no_augment):
            # augment by a random rotation
            rot_degrees = self.random.choice([0, 90, 180, 270])
            X_batch = rotate_pts(X_batch, degrees=rot_degrees)
            if not self.y_counts_only:
                Y_batch = rotate_pts(Y_batch, degrees=rot_degrees)
            # random gaussian noise
            if ARGS.noise_sigma is not None:
                # convert meters to 0-1 scale
                sigma = scale_meters_to_0_1(ARGS.noise_sigma, subdivide=self.n_subdivide)
                X_batch += self.random.normal(
                                        loc=0, 
                                        scale=sigma,
                                        size=self.x_batch_shape)

        x = tf.constant(X_batch, dtype=K.floatx())
        y = tf.constant(Y_batch, dtype=K.floatx())

        self.batch_time += time.perf_counter() - t1
        if return_ids:
            return x, y, self.valid_patch_ids[idx:end_idx]
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

    def get_patch(self, region, patch_num, subdiv_x, subdiv_y):
        """
        get the subdivided patch id in the dataseteeeeee
        args:
            region: str, region name
            patch_num: int
            subdiv_x/y: x/y index coordinates of subpatch (zero-based)
        returns:
            X data, Y data, patch_id
        """
        # set temporary sorted ids
        old_training = self.training
        old_ids = self.valid_patch_ids
        old_batchsize = self.batch_size
        self.training = False
        self.sorted()
        self.batch_size = 1
        # find batch index
        key = (region, patch_num, subdiv_x, subdiv_y)
        index = self.valid_patch_ids.index(key)
        # load batch
        X, Y = self[index]
        # restore correct values
        self.valid_patch_ids = old_ids
        self.batch_size = old_batchsize
        self.training = old_training
        return X[0], Y[0], key

    def get_naip(self, patch_id):
        """
        get naip image. Channels: R-G-B-NIR. Min 0, max 1.
        args:
            patch_id: id for subdiv or full patch. tuple: (region, patch_num, ...)
        """
        bounds = self.get_patch_bounds(patch_id)
        # load naip file
        region, patch_num = patch_id[:2]
        return load_naip(region, patch_num, bounds=bounds)


    def get_patch_bounds(self, patch_id):
        """
        args:
            patch_id: tuple: (region, patch_num[, ...])
        """
        try:
            return self.bounds_subdiv[patch_id]
        except KeyError:
            return self.bounds_full[patch_id]

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
        min_xyz, max_xyz = self.get_norm_data(patch_id)
        n_channels = pts.shape[-1]
        min_xyz = min_xyz[:n_channels]
        max_xyz = max_xyz[:n_channels]
        return (pts * (max_xyz - min_xyz)) + min_xyz

    def get_norm_data(self, patch_id):
        """
        get the data to transform georeferenced points to 0-1 scale, or vice versa
        returns:
            norm_mins: min values for each channel (x, y, z, R, G, B, NIR, NDVI)
            norm_maxs: max values
        """
        bound = self.get_patch_bounds(patch_id)
        min_x,min_y,max_x,max_y = bound.minmax_fmt()

        # spatial channels, then spectral channels.
               # x     y      z              r  g  b  nir ndvi
        mins = [min_x, min_y, 0]          + [0, 0, 0, 0, -1.0]
        maxs = [max_x, max_y, Z_MAX_CLIP] + [1, 1, 1, 1, 1.0]

        mins = np.array(mins)[self.channel_mask]
        maxs = np.array(maxs)[self.channel_mask]

        return mins, maxs


    def summary(self):
        print("LidarPatchGen '{}' from dataset '{}' regions:".format(self.name, self.dsname),  ", ".join(self.regions))
        print(" ", self.num_ids, "patches, in", len(self), "batches, batchsize", self.batch_size)
        print(" ", self.npoints, "points per patch.")
        total_pts = sum([x.shape[0] for x in self.lidar_subdiv.values()])
        print(" ", "{:.3e}".format(total_pts), "potential input points in dataset (not all get used every epoch)")
        total_trees = [x.shape[0] for x in self.gt_subdiv.values()]
        print(" ", sum(total_trees), "total trees. ({:.2f} avg, {:.2f} median per patch)".format(sum(total_trees) / self.num_ids, np.median(total_trees)))
        min_trees = min([x.shape[0] for x in self.gt_subdiv.values()])
        max_trees = max([x.shape[0] for x in self.gt_subdiv.values()])
        print(" ", "min trees:", min_trees, "max trees:", max_trees)
        if ARGS.handle_small == "drop":
            print("   ", self.num_small_patches, "patches dropped for having too few lidar points")
        else:
            print(" ", self.num_small_patches, "patches had too few points, were filled/repeated")
            print("   ", self.num_pts_filled, "total pts added in patches")
            print("   ", "max points added:", self.max_pts_filled, "in", self.smallest_patch)
        try:
            print("  xbatch shape:", self.x_batch_shape)
            print("  ybatch shape:", self.y_batch_shape)
        except AttributeError:
            pass

    def on_epoch_end(self):
        print("avg batch time:", self.batch_time / len(self))
        print("total batch time:", self.batch_time)
        self.batch_time = 0
        self.random.shuffle(self.valid_patch_ids)

    def sorted(self):
        """put ids in a reproducable order (sorted order)"""
        self.valid_patch_ids.sort()


def get_datasets(dsname, regions, sets=("train", "val", "test"), batchsize=None):
    """
    get the `sets` datasets from ds `dsname`
    returns:
        list of LidarPatchGen
    """
    timer = MyTimer()
    splits = get_data_splits(sets=sets, regions=regions)

    result = []
    for name, patch_ids in zip(sets, splits):
        if name == "train":
            train_gen = LidarPatchGen(patch_ids, name="train", training=True, batchsize=batchsize)
            result.append(train_gen)
        elif name == "val":
            val_gen = LidarPatchGen(patch_ids, name="validation", batchsize=batchsize)
            result.append(val_gen)
        elif name == "test":
            test_gen = LidarPatchGen(patch_ids, name="test", batchsize=batchsize)
            result.append(test_gen)
        else:
            raise ValueError("Unknown ds set '{}'".format(name))
    
    timer.measure("dataset generation")

    return result




if __name__ == "__main__":
    ARGS.dsname = "nohagclip"
    ARGS.handle_small = "fill"
    ARGS.batchsize = 16
    ARGS.subdivide = 5
    ARGS.loss = "mmd"
    ARGS.npoints = 800
    ARGS.noise_sigma = 0.0
    ARGS.test = False

    regions = get_all_regions(ARGS.dsname)

    train_gen, val_gen, test_gen = get_datasets(ARGS.dsname, regions)
    train_gen.summary()
    val_gen.summary()
    test_gen.summary()

    X, Y = train_gen.load_all()

    print(X.numpy().max(axis=0).max(axis=0))
    print(X.numpy().min(axis=0).min(axis=0))

    print(Y.numpy().max(axis=0).max(axis=0))
    print(Y.numpy().min(axis=0).min(axis=0))

    X, Y = val_gen.load_all()

    print(X.numpy().max(axis=0).max(axis=0))
    print(X.numpy().min(axis=0).min(axis=0))

    print(Y.numpy().max(axis=0).max(axis=0))
    print(Y.numpy().min(axis=0).min(axis=0))
        

