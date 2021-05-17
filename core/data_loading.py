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

    def __init__(self, filename, skip_freq=None, keep_freq=None):
        """
        args:
            skip_freq: every skip_freq patches will be skipped (designated for validation)
            keep_freq: every keep_freq patches will be kept (designated for validation) 
                (only one of skip_freq and keep_freq should be provided)
        """
        assert not (skip_freq is not None and keep_freq is not None)
        self.batch_size = args.batchsize
        self.file = h5py.File(filename, "r")
        atexit.register(self.close_file) # close file on completion
        self.init_data(skip_freq, keep_freq)
        self.y_counts_only = False
        if args.mode == "count":
            self.y_counts_only = True

    def init_data(self, skip_freq, keep_freq):
        self.max_points = self.file["lidar"].attrs["max_points"]
        self.min_points = self.file["lidar"].attrs["min_points"]
        self.max_trees = self.file["gt"].attrs["max_trees"]
        self.min_trees = self.file["gt"].attrs["min_trees"]
        all_ids = list(self.file['lidar'].keys())
        self.num_ids = len(all_ids)
        if skip_freq is not None:
            all_ids = set(all_ids) - set(all_ids[::skip_freq])
            all_ids = list(all_ids)
        elif keep_freq is not None:
            all_ids = all_ids[::keep_freq]
        self.ids = all_ids
        np.random.shuffle(self.ids)

    def __len__(self):
        return self.num_ids // self.batch_size

    def __getitem__(self, idx):
        end_idx = idx + self.batch_size
        x = []
        y = []
        for i in self.ids[idx:end_idx]:
            x.append(self.file['lidar/'+i][:])
            if self.y_counts_only:
                y.append(self.file['gt/'+i].shape[0])
            else:
                y.append(self.file['gt/'+i][:])
        if args.ragged:
            x = tf.ragged.constant(x, ragged_rank=1, inner_shape=(3,), dtype=tf.float32)
        else:
            # x = [i.tolist() for i in x]
            x = [i[:self.min_points] for i in x]
            x = tf.constant(x, dtype=tf.float32)
        y = tf.constant(y, dtype=tf.float32)
        return x, y

    def load_all(self):
        """
        load all examples into one big array
        """
        pass

    def on_epoch_end(self):
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




def make_data_generators(val_split=0.1, val_as_gen=True):
    """
    returns:
        train Keras Sequence, val Sequence or raw data, test Sequence
    """
    val_freq = int(1/val_split)

    train_gen = LidarPatchGen("data/patches.h5", skip_freq=val_freq)
    val_gen = LidarPatchGen("data/patches.h5", keep_freq=val_freq)
    if not val_as_gen:
        val_gen = val_gen.load_all()
    # test_gen = LidarPatchGen(None)
    test_gen = None


    return train_gen, val_gen, test_gen



if __name__ == "__main__":
    tr, v, te = make_data_generators("count", 32)
