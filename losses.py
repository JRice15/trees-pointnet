import h5py
import keras
import numpy as np
import tensorflow as tf
from keras import Model
from keras import backend as K
from keras import layers
from keras.optimizers import Adam


def get_loss(mode, args):
    mode = mode.lower()
    if mode == "pointwise-treetop":
        return pointwise_dist_sqr(args)


    raise ValueError("No loss for mode '{}'".format(name))



def pointwise_dist_sqr(args):
    """
    squared xy distance of each point to closest ground truth target, weighted by 
    """
    dist_weight = args.dist_weight
    def loss(x, y):
        x_locs, gt = y
        



