import argparse
from pprint import pprint

import h5py
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras import backend as K
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from tensorflow import keras

from losses import get_loss
from models import pointnet
from dataset import make_data_generators

print("TF version:", tf.__version__)
print("Keras version:", keras.__version__)

parser = argparse.ArgumentParser()
parser.add_argument("--mode",required=True)
parser.add_argument("--epochs",type=int,default=100)
parser.add_argument("--batchsize",type=int,default=3)
parser.add_argument("--dist-weight",type=float,default=0.5,help="pointnet-treetop mode: weight on distance vs sum loss")
args = parser.parse_args()
pprint(vars(args))


if args.mode in ["pointwise-treetop"]:
    args.output_type = "seg"
elif args.mode in ["count"]:
    args.output_type = "cls"
else:
    raise ValueError("unknown mode to outputtype initialization")

output_features_map = {
    "pointwise-treetop": 1,
    "count": 1,
}

model = pointnet(
    args, 
    nattributes=3, 
    output_features=output_features_map[args.mode]
)
# model.summary()
# tf.keras.utils.plot_model(model)

loss, metrics = get_loss(args)

model.compile(
    loss=loss, 
    metrics=metrics,
    optimizer=Adam(0.0001)
)

# x = [
#     [[1, 2, 3], [2, 3, 4], [3, 4, 5]],
#     [[1, 2, 3], [2, 3, 4]],
#     [[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6], [5, 6, 7]],
# ]
# x = tf.ragged.constant(x, ragged_rank=1, dtype=tf.float32)

# y = [
#     [[1, 2], [-100, -100]],
#     [[3, 3], [3, 3]],
#     [[2, 2], [4, 4]]
# ] # cls
# y = tf.constant(y, dtype=tf.float32)

# y = [
#     [2], 
#     [1], 
#     [3],
# ]
# y = tf.constant(y, dtype=tf.float32)

# # below code is not supported until tf2.5 probably
# if args.output_type == "seg":
#     y = [
#         [[1, 2, 3], [2, 3, 4], [3, 4, 5]],
#         [[1, 2, 3], [2, 3, 4]],
#         [[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6], [5, 6, 7]],
#     ] # seg
#     y = tf.ragged.constant(y, ragged_rank=1, dtype=tf.float32)


train_gen, val_gen, test_gen = make_data_generators(args.mode, args.batchsize, val_split=0.1, test_split=0.1)

# they dont support sequences as val data for some reason
valx, valy = [], []
for i in range(len(val_gen)):
    x, y = val_gen[i]
    print(x)
    print(y)
    valx += [np.array(i).astype(np.float32) for i in x]
    valy += y

print(valx)
valx = tf.ragged.constant([tf.constant(i) for i in valx], ragged_rank=1)

valy = tf.constant(valy)


model.fit(
    x=train_gen, 
    validation_data=(valx, valy),
    epochs=args.epochs, 
    batch_size=args.batchsize
)


