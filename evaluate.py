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

#metrics = [v for i,v in enumerate(model.metrics) if model.metrics_names[i] != "loss"]
metrics = [
    "mean_squared_error",
    "mean_absolute_error",
    "RootMeanSquaredError",
    "mean_absolute_percentage_error",
]

print(">>>>>>>")
print(model.metrics)
print(model.metrics_names)

model.compile(
    optimizer=model.optimizer,
    loss=model.loss,
    metrics=metrics
)

test_gen = data_loading.get_test_gen()
test_gen.summary()

metric_vals = model.evaluate(test_gen)

results = {model.metrics_names[i]:v for i,v in enumerate(metric_vals)}

print() # newline
for k,v in results.items():
    print(k+":", v)

with open(os.path.join(RESULTS_DIR, "results.json"), "w") as f:
    json.dump(results, f, indent=2)

print()
x, y = test_gen.load_all()
predictions = model.predict(x)

print("First 10 predictions, ground truths:")
for i in range(10):
    print(" ", np.squeeze(predictions[i]), y[i].numpy())
