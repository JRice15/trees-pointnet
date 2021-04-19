import keras
import numpy as np
import tensorflow as tf
from keras import Model
from keras import backend as Kbackend
from keras import layers
from keras.optimizers import Adam

from models import pointnet


print("TF version:", tf.__version__)
print("Keras version:", keras.__version__)

model = pointnet(10, 3)

# model.summary()
keras.utils.plot_model(model)



model.compile(loss=keras.losses.mse, optimizer=Adam())

x = np.random.rand(1, 10, 3)
y = np.random.rand(1, 10, 50)

model.fit(x, y)

out = model(x)

print(out.shape)