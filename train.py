import keras
import numpy as np
import tensorflow as tf
from keras import Model
from keras import backend as Kbackend
from keras import layers
from keras.optimizers import Adam

from models import pointnet

tf.config.experimental_run_functions_eagerly(True)

print("TF version:", tf.__version__)
print("Keras version:", keras.__version__)

model = pointnet(10, 3)

model.compile()

x = np.random.rand(1, 10, 3)
y = np.random.rand(1, 10, 3)

out = model(x)

print(out.shape)