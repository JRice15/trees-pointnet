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
parser.add_argument("--epochs",type=int,default=100)
parser.add_argument("--batchsize",type=int,default=3)
parser.add_argument("--dist-weight",type=float,help="pointnet-treetop mode: weight on distance vs sum loss")
args = parser.parse_args()
pprint(vars(args))


model = pointnet(args.mode, nattributes=3, output_features=3, batchsize=args.batchsize)
model.summary()
# keras.utils.plot_model(model)

class RaggedMSE(keras.losses.Loss):
    @tf.function
    def call(self, y_true, y_pred):
        losses = tf.ragged.map_flat_values(
            keras.losses.mse, y_true, y_pred)
        return tf.reduce_mean(losses)


def loss(a, b):
    return K.sum(a - b)

model.compile(
    # loss=get_loss(args.mode), 
    loss=keras.losses.mse,
    # loss=RaggedMSE(),
    # loss=loss,
    metrics=["mse"],
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
] # cls
# y = [
#     [[1, 2, 3], [2, 3, 4], [3, 4, 5]],
#     [[1, 2, 3], [2, 3, 4]],
#     [[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6], [5, 6, 7]],
# ] # seg

x = tf.ragged.constant(x, ragged_rank=1, dtype=tf.float32)
y = tf.constant(y, dtype=tf.float32)
# y = tf.ragged.constant(y, ragged_rank=1, dtype=tf.float32)

# x = np.random.random((1,10,3))
# y = np.random.random((1,1))

model.fit(x, y, epochs=args.epochs, batch_size=args.batchsize)

# out = model(x)

# print(out.shape)

