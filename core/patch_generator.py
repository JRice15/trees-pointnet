import json
import os
import glob
import sys
import time
from pathlib import PurePath

import h5py
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model
from tensorflow.keras import backend as K
from tensorflow.keras import layers

from core import ARGS, REPO_ROOT
from core.utils import raster_plot


def rotate(p, degrees=0):
    """
    in-place rotate points `p` counterclockwise by a multiple of 90 degrees, 
    around the point (0.5, 0.5)
    """
    if degrees == 0:
        return p
    origin = np.zeros_like(p)
    origin[...,:2] = 0.5
    p -= origin
    assert degrees % 90 == 0
    if degrees == 180:
        p[...,:2] = -p[...,:2]
    else:
        p[...,:2] = p[..., 1::-1]
        if degrees == 90:
            p[...,1] = -p[...,1]
        else:
            p[...,0] = -p[...,0]
    p += origin
    return p




class LidarPatchGen(keras.utils.Sequence):
    """
    loads data from a patches h5 file.
    The file should have this structure:

    / [attributes: gridrows, gridcols, gridsize, grid_min_x, grid_min_y]
    /gt: group [attributes: min_trees, max_trees]
    /lidar: group [attributes: min_points, max_points]
    /gt/patchX_Y: dataset, patch from grid column X row Y, shape (numtrees, 2)
        channels: x, y
    /lidar/patchX_Y: dataset, patch from grid column X row Y, shape (numpts, 4)
        channels: x, y, height above ground, ndvi
    /naip/patchX_Y: dataset, square NAIP image from grid column X row Y, shape ~(128,128)
    """

    def __init__(self, patch_ids, dataset_dir, name=None, batchsize=None, training=False):
        """
        patch_ids: list(tuple), where each tuple is of the format (filename, patch number)
        in the ARGS global object:
            mode
            batchsize
            npoints: points to input per patch
            ndvi: bool, whether to include ndvi channel
            ragged: bool
            training: whether this gen is being used for training (will augment data)
        """
        self.name = name
        self.dsname = ARGS.dsname
        self.batch_size = ARGS.batchsize if batchsize is None else batchsize
        self.use_ndvi = ARGS.ndvi
        self.patch_ids = np.array(patch_ids)
        self.dataset_dir = dataset_dir
        self.regions = list(set([i[0] for i in self.patch_ids]))
        self.files = {
            f:h5py.File(dataset_dir.joinpath(f+".h5").as_posix(), "r") for f in self.regions
        }
        self.y_counts_only = False
        if ARGS.loss == "count":
            self.y_counts_only = True
        self.nattributes = 3 + (1 if self.use_ndvi else 0)
        self.npoints = ARGS.npoints
        self.z_max = 50 # for normalization
        self.init_rng()
        self.init_data()
        self.batch_time = 0
        self.training = training

        if ARGS.ragged:
            self._f_getitem = self._ragged_getitem
        else:
            self._f_getitem = self._nonrag_getitem

    def init_rng(self):
        """initialize or reinitialize the random number generatore"""
        self.random = np.random.default_rng(seed=44)

    def init_data(self):
        # get max gt trees
        maxtrees = 0
        for file in self.files.values():
            keys = file["gt"].keys()
            maxtrees = max(
                maxtrees, 
                max([file["gt/"+k].shape[0] for k in keys])
            )
        self.max_trees = maxtrees

        # filter patch ids
        orig_num_ids = len(self.patch_ids)
        self.patch_ids = np.array([
            i for i in self.patch_ids
            if self.files[i[0]]["/lidar/"+i[1]].shape[0] >= self.npoints
        ])
        self.num_ids = len(self.patch_ids)
        self.num_filtered_ids = orig_num_ids - self.num_ids
        if ARGS.test:
            self.num_ids = 2

        # get normalization data
        self.norm_data = {}
        for region,file in self.files.items():
            min_xyz = []
            max_xyz = []
            # x
            x = file["grid"][:,[0,2]]
            min_xyz.append(x.min(axis=-1))
            max_xyz.append(x.max(axis=-1))
            # y
            y = file["grid"][:,[1,3]]
            min_xyz.append(y.min(axis=-1))
            max_xyz.append(y.max(axis=-1))
            # z, varies from 0 to custom z_max
            gridlen = len(x)
            min_xyz.append(np.zeros(gridlen))
            max_xyz.append(np.ones(gridlen) * self.z_max)
            # ndvi, varies from -1 to 1
            if self.use_ndvi:
                min_xyz.append(-np.ones(gridlen))
                max_xyz.append(np.ones(gridlen))
            min_xyz = np.array(min_xyz).T
            max_xyz = np.array(max_xyz).T
            self.norm_data[region] = {
                "min_xyz": min_xyz,
                "max_xyz": max_xyz
            }

        # sort for reproducibility
        self.sorted()
        self.random.shuffle(self.patch_ids)
        if not ARGS.ragged:
            self._x_batch = np.empty((self.batch_size, self.npoints, self.nattributes))
            if self.y_counts_only:
                self._y_batch = np.empty((self.batch_size,))
            else:
                self._y_batch = np.empty((self.batch_size, self.max_trees, 3)) # 3 -> (x,y,isvalid)

    def __len__(self):
        return self.num_ids // self.batch_size

    def __getitem__(self, idx, return_ids=False):
        return self._f_getitem(idx, return_ids=return_ids)

    def _nonrag_getitem(self, idx, return_ids=False):
        """
        __getitem__ for nonragged outputs
        """
        t1 = time.perf_counter()
        idx = idx * self.batch_size
        end_idx = idx + self.batch_size
        self._y_batch.fill(0)
        for i,(region,patchname) in enumerate(self.patch_ids[idx:end_idx]):
            patch_num = int(patchname[5:]) # remove "patch" prefix from "patchNNN" to get patch number
            file = self.files[region]
            # select <npoints> evenly spaced points randomly from batch.
            #   this is done because indices must be in ascending order, and indexing with an arbitrary array is 
            #   orders of magnitude slower than a simple slice
            x_node = file['lidar/'+patchname]
            num_x_pts = x_node.shape[0]
            step = num_x_pts // self.npoints
            leftover = num_x_pts % self.npoints
            if leftover == 0:
                rand_offset = 0
            else:
                rand_offset = self.random.choice(leftover)
            top_offset = leftover - rand_offset
            # get data from file
            self._x_batch[i] = file['lidar/'+patchname][rand_offset:num_x_pts-top_offset:step, :self.nattributes]

            # normalize data
            min_xyz = self.norm_data[region]["min_xyz"][patch_num]
            max_xyz = self.norm_data[region]["max_xyz"][patch_num]
            self._x_batch[i] = (self._x_batch[i] - min_xyz) / (max_xyz - min_xyz)
            
            # select all gt y points, or just y count
            if self.y_counts_only:
                self._y_batch[i] = file['gt/'+patchname].shape[0]
            else:
                min_xy, max_xy = min_xyz[:2], max_xyz[:2]
                ydata = file['gt/'+patchname][:]
                self._y_batch[i,:ydata.shape[0],2] = 1
                self._y_batch[i,ydata.shape[0]:,2] = 0
                self._y_batch[i,:ydata.shape[0],:2] = (ydata - min_xy) / (max_xy - min_xy)

        # shuffle input points within each patch
        #  this shuffles the points within each seperate patch in the same way, but that is random enough for me
        self.random.shuffle(self._x_batch, axis=1)

        if self.training:
            # augment by a random rotation
            rot_degrees = self.random.choice([0, 90, 180, 270])
            self._x_batch = rotate(self._x_batch, degrees=rot_degrees)
            if not self.y_counts_only:
                self._y_batch = rotate(self._y_batch, degrees=rot_degrees)
            # random gaussian noise
            if ARGS.noise_sigma is not None:
                self._x_batch += self.random.normal(
                                        loc=0, 
                                        scale=ARGS.noise_sigma, 
                                        size=self._x_batch.shape)

        x = tf.constant(self._x_batch, dtype=tf.float32)
        y = tf.constant(self._y_batch, dtype=tf.float32)
        self.batch_time += time.perf_counter() - t1
        if return_ids:
            return x, y, self.patch_ids[idx:end_idx]
        return x, y

    def _ragged_getitem(self, idx):
        """
        __getitem__ for ragged outputs
        PROBABLY DOESNT WORK ANYMORE, I haven't kept the ragged code up to date
        since it is much slower and less effective
        """
        raise NotImplementedError()
        t1 = time.perf_counter()
        idx = idx * self.batch_size
        end_idx = idx + self.batch_size
        x = []
        y = []
        for i in self.patch_ids[idx:end_idx]:
            x.append(self.file['lidar/'+i][:])
            if self.y_counts_only:
                y.append(self.file['gt/'+i].shape[0])
            else:
                y.append(self.file['gt/'+i][:])
        x = tf.ragged.constant(x, ragged_rank=1, inner_shape=(self.nattributes,), dtype=tf.float32)
        y = tf.constant(y, dtype=tf.float32)
        self.batch_time += time.perf_counter() - t1
        return x, y

    def load_all(self):
        """
        load all examples into one big array. should be used for the validation
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

    def get_patch(self, i=None, region=None):
        """
        get the full i'th patch of the entire sorted dataset, or from a specific region
        args:
            i: int
            region: str, region name
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
        # get results
        if region is None:
            id = self.patch_ids[i]
            x, y = self[i]
        else:
            id = (region, "patch"+str(i))
            index = self.patch_ids.index(id)
            x, y = self[index]
        # restore correct values
        self.patch_ids = old_ids
        self.batch_size = old_batchsize
        self.training = old_training
        return x[0], y[0], id

    def get_naip(self, patch_id):
        """
        get naip image. Channels: R-G-B-NIR
        args:
            patch_id: tuple: (region name str, patch id str of the form "patch322")
        """
        file, patchname = patch_id
        return self.files[file]["/naip/"+patchname][:]

    def get_batch_shape(self):
        """
        get x,y shape of one batch
        x: (batchsize; npoints or None if ragged; nattributes per point)
        """
        x, y = self[0]
        print("Batch shape x:", x.shape, "y:", y.shape)
        print(x)
        print(y)
        return x.shape, y.shape

    def summary(self):
        print("LidarPatchGen '{}' from dataset '{}' regions:".format(self.name, self.dsname),  ", ".join(self.regions))
        print(" ", self.num_ids, "patches, in", len(self), "batches, batchsize", self.batch_size)
        print(" ", self.npoints, "points per patch.", self.num_filtered_ids, "patches dropped for having too few lidar points")
        try:
            print("  xbatch shape:", self._x_batch.shape)
            print("  ybatch shape:", self._y_batch.shape)
        except AttributeError:
            pass

    def on_epoch_end(self):
        print("avg batch time:", self.batch_time / len(self))
        print("total batch time:", self.batch_time)
        self.batch_time = 0
        self.random.shuffle(self.patch_ids)

    def sorted(self):
        """put ids in a reproducable order (sorted order)"""
        ids = self.patch_ids
        sorting = np.lexsort((ids[:,1],ids[:,0]))
        self.patch_ids = ids[sorting]

    def __del__(self):
        for file in self.files.values():
            try:
                file.close()
            except:
                pass # already closed, that's fine




def get_tvt_split(dataset_dir, regions, val_step, test_step):
    train = []
    val = []
    test = []
    for region in regions:
        regionfile = dataset_dir.joinpath("{}.h5".format(region)).as_posix()
        with h5py.File(regionfile, "r") as f:
            patches = sorted(f["lidar"].keys())
        patches = [(region, x) for x in patches]
        test += patches[::test_step]
        rest_patches = [x for x in patches if x not in test]
        val += rest_patches[::val_step]
        train += [x for x in rest_patches if x not in val]
    return train, val, test


def get_train_val_gens(dataset_dir, regions, val_split=0.1, test_split=0.1,
        val_batchsize=None):
    """
    returns:
        train Keras Sequence, val Sequence or raw data, test Sequence
    """
    val_step = int(1/val_split)
    test_step = int(1/val_split)
    train, val, test = get_tvt_split(dataset_dir, regions, val_step, test_step)

    train_gen = LidarPatchGen(train, dataset_dir, name="train", training=True)
    val_gen = LidarPatchGen(val, dataset_dir, name="validation", batchsize=val_batchsize)

    return train_gen, val_gen

def get_test_gen(dataset_dir, regions, val_split=0.1, test_split=0.1):
    val_step = int(1/val_split)
    test_step = int(1/val_split)
    train, val, test = get_tvt_split(dataset_dir, regions, val_step, test_step)

    test_gen = LidarPatchGen(test, dataset_dir, name="test", batchsize=1)
    return test_gen



if __name__ == "__main__":
    train_gen, val_gen = get_train_val_gens()
    test_gen = get_test_gen()
    train_gen.summary()
    val_gen.summary()
    test_gen.summary()
