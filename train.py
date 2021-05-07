import argparse
from pprint import pprint
import time


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
import data_loading

print("TF version:", tf.__version__)
print("Keras version:", keras.__version__)

parser = argparse.ArgumentParser()
parser.add_argument("--mode",required=True)
parser.add_argument("--epochs",type=int,default=100)
parser.add_argument("--batchsize",type=int,default=16)
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
    optimizer=Adam(0.003)
)

# get data generators

train_gen, val_gen, test_gen = data_loading.make_data_generators(args.mode, 
                                    args.batchsize, val_split=0.1, test_split=0.1)



# model.fit(
#     # x=data_loading.generator_wrapper(args.mode, args.batchsize)(),
#     # x=data_loading.dataset_wrapper(args.mode, args.batchsize),
#     x=train_gen,
#     epochs=args.epochs,
#     batch_size=args.batchsize,
# )
# exit()


"""
adapted from https://keras.io/guides/writing_a_training_loop_from_scratch/
"""

@tf.function
def train_step(x, y):
    with tf.GradientTape() as tape:
        logits = model(x, training=True)
        loss_values = model.loss(y, logits)
        loss_values += model.losses
    grads = tape.gradient(loss_values, model.trainable_weights)
    model.optimizer.apply_gradients(zip(grads, model.trainable_weights))
    for m in model.metrics:
        m.update_state(y, logits)
    return loss_values

@tf.function
def test_step(x, y):
    logits = model(x, training=False)
    loss_values = model.loss(y, logits)
    loss_values += np.sum(model.losses)
    for m in model.metrics:
        m.update_state(y, logits)
    return loss_values

def eval_metrics():
    metric_vals = [m.result() for m in model.metrics]
    return ", ".join(["{} {:.5f}".format(name, val) for name, val in zip(model.metrics_names, metric_vals)])

for epoch in range(args.epochs):
    print("Epoch {}".format(epoch))
    start_time = time.time()

    pre = time.time()
    # Iterate over the batches of the dataset.
    for step, (x_batch_train, y_batch_train) in enumerate(train_gen):
        t1 = time.time()
        print("  > other:", t1-pre)
        # loss_values = train_step(x_batch_train, y_batch_train)
        losses = model.train_on_batch(x_batch_train, y_batch_train)
        print("  > step time:", time.time()-t1)

        pre = time.time()
        # Log every n batches.
        # if step % 20 == 0:
        #     print("   {:>3d} -- loss {:.5f}, {}".format(step, np.mean(loss_values), eval_metrics()))

    # Train metrics
    print("Train -- loss {:.5f}, {}".format(np.mean(loss_values), eval_metrics()))
    for m in model.metrics:
        m.reset_states()
        
    # Validation
    for x_batch_val, y_batch_val in val_gen:
        val_loss = test_step(x_batch_val, y_batch_val)
    print("Val   -- loss {:.5f}, {}".format(np.mean(val_loss), eval_metrics()))
    for m in model.metrics:
        m.reset_states()

    print("  Time taken: %.2fs" % (time.time() - start_time))



