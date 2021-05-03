import atexit

import h5py
import keras
import numpy as np
import tensorflow as tf
from keras import Model
from keras import backend as K
from keras import layers


class Dataset(keras.utils.Sequence):
    """
    loads data from 'data/patches.h5'
    """

    def __init__(self, batchsize, split_low=None, split_high=None):
        self.batchsize = batchsize
        self.file = h5py.File("data/patches.h5", "r")
        atexit.register(self.close) # close file on completion

        self.max_points = self.file["lidar"].attrs["max_points"]
        self.max_trees = self.file["gt"].attrs["max_trees"]

        all_ids = np.array(self.file.keys())
        num_ids = len(all_ids)
        if split_low is None:
            split_high = round(split_high * num_ids)
            self.ids = ids[:split_high]
        elif split_high is None:
            split_low = round(split_low * num_ids)
            self.ids = ids[split_low:]
        else:
            split_high = round(split_high * num_ids)
            split_low = round(split_low * num_ids)
            self.ids = ids[split_low:split_high]
        np.random.shuffle(self.ids)


    def __len__(self):
        return len(self.ids) / self.batchsize

    def __getitem__(self):
        X = np.zeros((batch_size,self.max_points,3))
        Y = np.zeros((batch_size,self.max_points,2))
        for i in range(batchsize):
            

    def x_shape(self):
        return (self.max_points, 3)

    def y_shape(self):
        return (self.max_trees, 2)

    def on_epoch_end(self):
        np.random.shuffle(self.ids)

    def close_file(self):
        try:
            self.file.close()
        except:
            pass # already closed, that's fine


def make_generators(batchsize, val_split=0.1, test_split=0.1):
    """returns train, val, test dataset generators"""
    val_low = -val_split - test_split
    test_low = -test_split
    
    train_gen = Dataset(batchsize, None, val_low)
    val_gen = Dataset(batchsize, val_low, test_low)
    test_gen = Dataset(batchsize, test_low, None)

    return train_gen, val_gen, test_gen

