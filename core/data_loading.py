import atexit
import time

import h5py
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras import backend as K
from tensorflow.keras import layers
from tensorflow import keras
from core import ARGS

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

    def __init__(self, filename, name=None, skip_freq=None, keep_freq=None, batchsize=None):
        """
        args:
            skip_freq: every skip_freq patches will be skipped (designated for validation)
            keep_freq: every keep_freq patches will be kept (designated for validation) 
                (only one of skip_freq and keep_freq should be provided)
        in the ARGS global object:
            mode
            batchsize
            npoints: points to input per patch
            ndvi: bool, whether to include ndvi channel
            ragged: bool
        """
        assert not (skip_freq is not None and keep_freq is not None)
        self.name = name
        self.batch_size = ARGS.batchsize if batchsize is None else batchsize
        self.use_ndvi = ARGS.ndvi
        self.filename = filename
        self.file = h5py.File(self.filename, "r")
        atexit.register(self.close_file) # close file on completion
        self.skip_freq = skip_freq
        self.keep_freq = keep_freq
        self.y_counts_only = False
        if ARGS.mode == "count":
            self.y_counts_only = True
        self.init_rng()
        self.init_data()
        self.batch_time = 0

    def init_rng(self):
        """initialize or reinitialize the random number generatore"""
        self.random = np.random.default_rng(seed=44)

    def init_data(self):
        self.max_points = self.file["lidar"].attrs["max_points"]
        self.min_points = self.file["lidar"].attrs["min_points"]
        # self.z_max = self.file["lidar"].attrs["z_max"]
        self.z_max = 50 # manual z max
        self.max_trees = self.file["gt"].attrs["max_trees"]
        self.min_trees = self.file["gt"].attrs["min_trees"]
        self.gridsize = self.file.attrs["gridsize"]
        self.grid_min_x = self.file.attrs["grid_min_x"]
        self.grid_min_y = self.file.attrs["grid_min_y"]
        self.nattributes = 3 + (1 if self.use_ndvi else 0)
        self.npoints = ARGS.npoints

        # load patch ids
        all_ids = list(self.file['lidar'].keys())
        if self.skip_freq is not None:
            all_ids = set(all_ids) - set(all_ids[::self.skip_freq])
            all_ids = list(all_ids)
        elif self.keep_freq is not None:
            all_ids = all_ids[::self.keep_freq]
        # filter patches with too few points
        len1 = len(all_ids)
        all_ids = [i for i in all_ids if self.file['lidar/'+i].shape[0] >= self.npoints]
        len2 = len(all_ids)
        self.num_patches_filtered = len1 - len2

        self.num_ids = len(all_ids)
        if ARGS.test:
            self.num_ids = 2
        # sort for reproducibility
        self.ids = np.array(sorted(all_ids))
        self.random.shuffle(self.ids)
        if not ARGS.ragged:
            self._x_batch = np.empty((self.batch_size, self.npoints, self.nattributes))
            if self.y_counts_only:
                self._y_batch = np.empty((self.batch_size,))
            else:
                self._y_batch = np.empty((self.batch_size, self.max_trees, 3)) # 3 -> (x,y,isvalid)

    def __len__(self):
        return self.num_ids // self.batch_size

    def __getitem__(self, idx, ):
        return self._ragged_getitem(idx) if ARGS.ragged else self._nonrag_getitem(idx)

    def _nonrag_getitem(self, idx):
        """
        __getitem__ for nonragged outputs
        """
        t1 = time.perf_counter()
        idx = idx * self.batch_size
        end_idx = idx + self.batch_size
        self._y_batch.fill(0)
        for i,patch in enumerate(self.ids[idx:end_idx]):
            # select <npoints> evenly spaced points randomly from batch
            x_node = self.file['lidar/'+patch]
            num_x_pts = x_node.shape[0]
            step = num_x_pts // self.npoints
            leftover = num_x_pts % self.npoints
            if leftover == 0:
                rand_offset = 0
            else:
                rand_offset = self.random.choice(leftover)
            top_offset = leftover - rand_offset
            # get data from file
            self._x_batch[i] = self.file['lidar/'+patch][rand_offset:num_x_pts-top_offset:step, :self.nattributes]
            
            # normalize data
            min_xy = np.amin(self._x_batch[i,:,:2], axis=0)
            max_xy = np.amax(self._x_batch[i,:,:2], axis=0)
            # z channel
            min_xyz = np.append(min_xy, 0)
            max_xyz = np.append(max_xy, self.z_max)
            if self.use_ndvi:
                # ndvi channel varies from -1 to 1
                min_xyz = np.append(min_xyz, -1)
                max_xyz = np.append(max_xyz, 1)
            self._x_batch[i] = (self._x_batch[i] - min_xyz) / (max_xyz - min_xyz)
            
            # select all gt y points, or just y count
            if self.y_counts_only:
                self._y_batch[i] = self.file['gt/'+patch].shape[0]
            else:
                ydata = self.file['gt/'+patch][:]
                self._y_batch[i,:ydata.shape[0],2] = 1
                self._y_batch[i,ydata.shape[0]:,2] = 0
                self._y_batch[i,:ydata.shape[0],:2] = (ydata - min_xy) / (max_xy - min_xy)

        x = tf.constant(self._x_batch, dtype=tf.float32)
        y = tf.constant(self._y_batch, dtype=tf.float32)
        self.batch_time += time.perf_counter() - t1
        return x, y

    def _ragged_getitem(self, idx):
        """
        __getitem__ for ragged outputs
        PROBABLY DOESNT WORK ANYMORE, I haven't kept the ragged code up to date
        since it is much slower and less effective
        """
        t1 = time.perf_counter()
        idx = idx * self.batch_size
        end_idx = idx + self.batch_size
        x = []
        y = []
        for i in self.ids[idx:end_idx]:
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

    def get_patch(self, *, i=None, xy=None):
        """
        get the full i'th patch, or patch{x}_{y}, from this dataset
        returns:
            x, y, patch_id
        """
        # set temporary sorted ids
        old_ids = self.ids
        self.ids = sorted(self.ids)
        old_batchsize = self.batch_size
        self.batch_size = 1
        # get results
        if i is None:
            id = self.ids[i]
            x, y = self[i]
        # restore correct values
        self.ids = old_ids
        self.batch_size = old_batchsize
        return x[0], y[0], id

    def get_naip(self, patchname):
        """
        get naip image. Channels: R-G-B-NIR
        """
        return self.file["/naip/{}".format(patchname)][:]

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
        print("Dataset", self.name, "from", self.filename)
        print("  skip_freq", self.skip_freq, "keep_freq", self.keep_freq)
        print(" ", self.num_ids, "patches, in", len(self), "batches, batchsize", self.batch_size)
        print("  raw patches:", self.min_points, "pts min,", self.max_points, "pts max")
        print(" ", self.npoints, "points per patch.", self.num_patches_filtered, "patches filtered for having too few points")
        try:
            print("  xbatch shape:", self._x_batch.shape)
            print("  ybatch shape:", self._y_batch.shape)
        except AttributeError:
            pass

    def on_epoch_end(self):
        print("avg batch time:", self.batch_time / len(self))
        self.batch_time = 0
        self.random.shuffle(self.ids)

    def sorted(self):
        """put ids in a reproducable order (sorted order)"""
        self.init_rng()
        self.ids = sorted(self.ids)

    def close_file(self):
        try:
            self.file.close()
        except:
            pass # already closed, that's fine




def generator_wrapper(sequence):
    """turn Keras Sequence to generator"""
    def gen():
        while True:
            for i in range(len(sequence)):
                yield sequence[i]
    return gen

def dataset_wrapper(sequence):
    """
    turn Keras Sequence to generator than tf.data.Dataset.
    There is probably a better way to achieve this?
    """
    if ARGS.ragged:
        xspec = tf.RaggedTensorSpec(shape=(ARGS.batchsize, None, 3), dtype=tf.float32),
    else:
        x,y = sequence[0]
        xspec = tf.TensorSpec(shape=x.shape, dtype=tf.float32)

    train_gen = tf.data.Dataset.from_generator(
        generator_wrapper(sequence),
        output_signature=(
            xspec,
            tf.TensorSpec(shape=(), dtype=tf.float32)
        )
    )
    return train_gen




def get_train_val_gens(val_split=0.1):
    """
    returns:
        train Keras Sequence, val Sequence or raw data, test Sequence
    """
    val_freq = int(1/val_split)

    train_gen = LidarPatchGen("data/train_patches.h5", name="train", skip_freq=val_freq)
    val_gen = LidarPatchGen("data/train_patches.h5", name="validation", keep_freq=val_freq)

    return train_gen, val_gen

def get_test_gen():
    test_gen = LidarPatchGen("data/test_patches.h5", name="test", batchsize=1)
    return test_gen



if __name__ == "__main__":
    train_gen, val_gen = get_train_val_gens()
    test_gen = get_test_gen()
    train_gen.summary()
    val_gen.summary()
    test_gen.summary()
