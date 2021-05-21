import contextlib
import datetime
import json
import os
import glob
import time

import h5py
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model
from tensorflow.keras import backend as K
from tensorflow.keras import callbacks, layers
from tensorflow.keras.optimizers import Adam

from core import DATA_DIR, OUTPUT_DIR, MAIN_DIR, args, data_loading
from core.losses import get_loss
from core.models import pointnet
from core.utils import MyModelCheckpoint, output_model


ALL_MODELS_DIR = os.path.join(MAIN_DIR, "models/")

matching_models = [i for i in glob.glob(os.path.join(ALL_MODELS_DIR, "*"+args.name, "model_*.tf"))]
if len(matching_models) > 1:
    print("Multiple models match 'name' argument:")
    print(" ", matching_models)
    print("Defaulting to the most recent:")
    # all the names begin with a date/time string, so sorting gives order by time
    matching_models.sort()
    MODEL_PATH = matching_models[-1]
    print(" ", MODEL_PATH)
    print("You can add the timestamp string to '--name' to specify a different model")
elif len(matching_models) == 0:
    print("No matching models!")
    exit()
else:
    MODEL_PATH = matching_models[0]
MODEL_DIR = os.path.dirname(MODEL_PATH)
RESULTS_DIR = os.path.join(MODEL_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# load original params into args object
params_file = os.path.join(MODEL_DIR, "params.json")
with open(params_file, "r") as f:
    params = json.load(f)
params.pop("name")
for k,v in params.items():
    setattr(args, k, v)


print("\nTesting phase")

# load best model
print("Loading model", MODEL_PATH)
model = keras.models.load_model(MODEL_PATH)

test_gen = data_loading.get_test_gen()
test_gen.summary()

results = model.evaluate(test_gen)

print() # newline
for i,v in enumerate(results):
    print(model.metrics_names[i] + ":", v)

x, y = test_gen.load_all()
predictions = model.predict(x)

print("First 10 predictions:")
print(predictions[:10])
print("First 10 Ground Truth")
print(y[:10].numpy())

