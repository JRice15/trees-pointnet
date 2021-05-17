import contextlib
import time

import h5py
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model
from tensorflow.keras import backend as K
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam

from core import args, data_loading
from core.losses import get_loss
from core.models import pointnet


# map modes to number of output features
output_features_map = {
    "pointwise-treetop": 1,
    "count": 1,
}

model = pointnet(
    args, 
    nattributes=3, 
    output_features=output_features_map[args.mode]
)
model.summary()
# tf.keras.utils.plot_model(model)

loss, metrics = get_loss(args)

model.compile(
    loss=loss, 
    metrics=metrics,
    optimizer=Adam(0.003)
)

# get data generators

train_gen, val_gen, test_gen = data_loading.make_data_generators(val_split=0.1, val_as_gen=args.ragged)


if not args.ragged:
    model.fit(
        # x=data_loading.generator_wrapper(args.mode, args.batchsize)(),
        # x=data_loading.dataset_wrapper(args.mode, args.batchsize),
        x=train_gen,
        validation_data=val_gen,
        epochs=args.epochs,
        batch_size=args.batchsize,
    )


else:
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

    LOG_FREQ = 20 # in batches

    for epoch in range(args.epochs):
        print("Epoch {}".format(epoch))
        start_time = time.time()

        step_time = 0
        batch_time = 0
        step_end_time = time.time()
        # Iterate over the batches of the dataset.
        for step, (x_batch_train, y_batch_train) in enumerate(train_gen):
            step_start_time = time.time()
            batch_time += step_start_time - step_end_time

            # loss_values = train_step(x_batch_train, y_batch_train)
            loss_values = model.train_on_batch(x_batch_train, y_batch_train)

            step_end_time = time.time()
            step_time += step_end_time - step_start_time
            # Log every n batches.
            if step % LOG_FREQ == 0:
                print("   {:>3d} -- avg time: step {:.3f}, batch {:.3f} -- loss {:.5f}, {}".format(
                    step, step_time/LOG_FREQ, batch_time/LOG_FREQ, np.mean(loss_values), eval_metrics()))
                step_time = 0
                batch_time = 0

        # Train metrics
        print("Train -- loss {:.5f}, {}".format(np.mean(loss_values), eval_metrics()))
        for m in model.metrics:
            m.reset_states()
            
        # Validation
        for x_batch_val, y_batch_val in val_gen:
            val_loss = model.test_on_batch(x_batch_val, y_batch_val)
        print("Val   -- loss {:.5f}, {}".format(np.mean(val_loss), eval_metrics()))
        for m in model.metrics:
            m.reset_states()

        print("  Time taken: %.2fs" % (time.time() - start_time))
        


