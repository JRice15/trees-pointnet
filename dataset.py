import atexit

import h5py
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras import backend as K
from tensorflow.keras import layers
from tensorflow import keras


class LidarPatchGen(keras.utils.Sequence):
    """
    loads data from 'data/patches.h5'
    """

    def __init__(self, mode, batch_size, split_low=None, split_high=None):
        self.batch_size = batch_size
        self.file = h5py.File("data/patches.h5", "r")
        atexit.register(self.close_file) # close file on completion
        self.init_data(split_low, split_high)
        self.y_counts_only = False
        # if mode == "count":
        #     self.y_counts_only = True

    def init_data(self, split_low, split_high):
        self.max_points = self.file["lidar"].attrs["max_points"]
        self.max_trees = self.file["gt"].attrs["max_trees"]
        all_ids = list(self.file['lidar'].keys())
        self.num_ids = len(all_ids)
        if split_low is None:
            split_high = round(split_high * self.num_ids)
            self.ids = all_ids[:split_high]
        elif split_high is None:
            split_low = round(split_low * self.num_ids)
            self.ids = all_ids[split_low:]
        else:
            split_high = round(split_high * self.num_ids)
            split_low = round(split_low * self.num_ids)
            self.ids = all_ids[split_low:split_high]
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
        idx = (idx + self.batch_size) % self.num_ids
        if idx < self.batch_size:
            # it looped around to the beginning, meaning the last loop might not have gotten enough pts
            np.random.shuffle(self.ids)
            for i in self.ids[0:idx]:
                x.append(self.file['lidar/'+i][:])
                if self.y_counts_only:
                    y.append(self.file['gt/'+i].shape[0])
                else:
                    y.append(self.file['gt/'+i][:])
        return x, y


    def close_file(self):
        try:
            self.file.close()
        except:
            pass # already closed, that's fine








def make_data_generators(mode, batchsize, val_split=0.1, test_split=0.1):
    """returns train, val, test dataset generators"""
    val_low = -val_split - test_split
    test_low = -test_split
    
    train_gen = LidarPatchGen(mode, batchsize, None, val_low)
    val_gen = LidarPatchGen(mode, batchsize, val_low, test_low)
    test_gen = LidarPatchGen(mode, batchsize, test_low, None)

    return train_gen, val_gen, test_gen



if __name__ == "__main__":
    tr, v, te = make_data_generators("count", 32)
