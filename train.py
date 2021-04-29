import argparse
from pprint import pprint

import h5py
import keras
import numpy as np
import tensorflow as tf
from keras import Model
from keras import backend as K
from keras import layers
from keras.optimizers import Adam

from models import pointnet

print("TF version:", tf.__version__)
print("Keras version:", keras.__version__)

parser = argparse.ArgumentParser()
parser.add_argument("--mode")
parser.add_argument("--dist-weight",type=float,help="pointnet-treetop mode: weight on distance vs sum loss")
args = parser.parse_args()
pprint(vars(args))


model = pointnet(args.mode, 3)
model.summary()
keras.utils.plot_model(model)


def loss(a, b):
    return K.sum(a - b)

model.compile(
    # loss=get_loss(args.mode), 
    loss=keras.losses.mse,
    # loss=loss,
    optimizer=Adam()
)

x = [
    [[1, 2, 3], [2, 3, 4], [3, 4, 5]],
    [[1, 2, 3], [2, 3, 4]],
    [[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6], [5, 6, 7]],
]
y = [
    [1, 2, 3],
    [1, 2, 3],
    [1, 2, 3],
]

x = tf.ragged.constant(x, ragged_rank=1)
y = tf.constant(y)

# x = np.random.random((1,10,3))
# y = np.random.random((1,1))

model.fit(x, y)

out = model(x)

print(out.shape)

