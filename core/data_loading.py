import atexit
import time

import h5py
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras import backend as K
from tensorflow.keras import layers
from tensorflow import keras
from core import args

class LidarPatchGen(keras.utils.Sequence):
    """
    loads data from a patches h5 file.
    The file should have this structure:

    /gt: group [attributes: min_trees, max_trees]
    /lidar: group [attributes: min_points, max_points]
    /gt/patchN: dataset N, shape (numtrees, 2)
    /lidar/patchN: dataset N, shape (numpts, 3)
    """

    def __init__(self, filename, skip_freq=None, keep_freq=None, batchsize=None):
        """
        args:
            skip_freq: every skip_freq patches will be skipped (designated for validation)
            keep_freq: every keep_freq patches will be kept (designated for validation) 
                (only one of skip_freq and keep_freq should be provided)
        """
        assert not (skip_freq is not None and keep_freq is not None)
        self.batch_size = args.batchsize if batchsize is None else batchsize
        self.filename = filename
        self.file = h5py.File(self.filename, "r")
        atexit.register(self.close_file) # close file on completion
        self.skip_freq = skip_freq
        self.keep_freq = keep_freq
        self.init_data()
        self.y_counts_only = False
        if args.mode == "count":
            self.y_counts_only = True
        self.batch_time = 0

    def init_data(self):
        self.max_points = self.file["lidar"].attrs["max_points"]
        self.min_points = self.file["lidar"].attrs["min_points"]
        self.max_trees = self.file["gt"].attrs["max_trees"]
        self.min_trees = self.file["gt"].attrs["min_trees"]
        self.npoints = args.npoints

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
        if args.test:
            self.num_ids = 2
        self.ids = np.array(all_ids)
        np.random.shuffle(self.ids)

    def __len__(self):
        return self.num_ids // self.batch_size

    def __getitem__(self, idx):
        t1 = time.time()
        idx = idx * self.batch_size
        end_idx = idx + self.batch_size
        x = []
        y = []
        for i in self.ids[idx:end_idx]:
            if args.ragged:
                x.append(self.file['lidar/'+i][:])
            else:
                x.append(self.file['lidar/'+i][:self.npoints])
            if self.y_counts_only:
                y.append(self.file['gt/'+i].shape[0])
            else:
                y.append(self.file['gt/'+i][:])
        if args.ragged:
            x = tf.ragged.constant(x, ragged_rank=1, inner_shape=(3,), dtype=tf.float32)
        else:
            x = tf.constant(x, dtype=tf.float32)
        y = tf.constant(y, dtype=tf.float32)
        self.batch_time += time.time() - t1
        return x, y

    def load_all(self):
        """
        load all examples into one big array. should only be used for the validation
        patch generator, because keras doesn't accept a Sequence for 
        """
        x_batches = []
        y_batches = []
        for i in range(len(self)):
            x, y = self[i]
            x_batches.append(x)
            y_batches.append(y)
        x = tf.concat(x_batches, axis=0)
        y = tf.concat(y_batches, axis=0)
        return x, y

    def get_batch_shape(self):
        """
        get shape of one batch:
        (batchsize, npoints or None if ragged, nattributes per point)
        """
        x, y = self[0]
        print("Batch shape:", x.shape)
        return x.shape

    def summary(self):
        print("Dataset", self.filename)
        print("  skip_freq", self.skip_freq, "keep_freq", self.keep_freq)
        print(" ", self.num_ids, "patches, in", len(self), "batches, batchsize", self.batch_size)
        print("  raw patches:", self.min_points, "pts min,", self.max_points, "pts max")
        print(" ", self.npoints, "points per patch.", self.num_patches_filtered, "patches filtered for having too few points")

    def on_epoch_end(self):
        print("avg batch time:", self.batch_time / len(self))
        self.batch_time = 0
        np.random.shuffle(self.ids)

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
    if args.ragged:
        xspec = tf.RaggedTensorSpec(shape=(args.batchsize, None, 3), dtype=tf.float32),
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




def get_train_val_gens(val_split=0.1, val_as_gen=True):
    """
    returns:
        train Keras Sequence, val Sequence or raw data, test Sequence
    """
    val_freq = int(1/val_split)

    train_gen = LidarPatchGen("data/train_patches.h5", skip_freq=val_freq)
    val_gen = LidarPatchGen("data/train_patches.h5", keep_freq=val_freq)

    return train_gen, val_gen

def get_test_gen():
    test_gen = LidarPatchGen("data/test_patches.h5", batchsize=2)
    return test_gen



if __name__ == "__main__":
    tr, v, te = make_data_generators("count", 32)
