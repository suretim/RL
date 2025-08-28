#!/usr/bin/env python3
# -*- coding: utf-8 -*-
\"\"\"Reptile-style self-supervised meta-learning on HVAC plant time series.
Usage:
 - Put your CSV files (temp,humid,light,ac,heater,dehum,hum) into ./lll_data
 - Run this script. It will train an encoder+predictor with Reptile meta-update.
 - The task is next-step prediction (self-supervised).
\"\"\"

import os
import glob
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import random

# -------------------- Config --------------------
DATA_DIR = \"./lll_data\"    # directory where your generated CSVs live
SEQ_LEN = 32                 # input window for each task
FEATURE_DIM = 7              # temp, humid, light, ac, heater, dehum, hum
HIDDEN_DIM = 64              # LSTM hidden dim
INNER_LR = 1e-2
META_LR = 1e-3
INNER_STEPS = 5
META_EPOCHS = 40
TASKS_PER_META = 16          # number of tasks sampled per meta-iteration
SEED = 42
BATCH_EVAL = 32              # how many held-out tasks to evaluate each epoch

np.random.seed(SEED)
tf.random.set_seed(SEED)
random.seed(SEED)

# -------------------- Data loader --------------------
def load_all_series(data_dir):
    files = sorted(glob.glob(os.path.join(data_dir, \"*.csv\")))
    series = []
    for f in files:
        try:
            arr = np.loadtxt(f, delimiter=',', skiprows=1)
            # Ensure shape (T, D)
            if arr.ndim == 1:
                # single-row CSV; skip
                continue
            # Keep only first FEATURE_DIM columns if there are extras
            if arr.shape[1] >= FEATURE_DIM:
                arr = arr[:, :FEATURE_DIM].astype(np.float32)
            else:
                # pad columns with zeros if fewer columns
                pad = np.zeros((arr.shape[0], FEATURE_DIM - arr.shape[1]), dtype=np.float32)
                arr = np.concatenate([arr.astype(np.float32), pad], axis=1)
            series.append(arr)
        except Exception as e:
            print(f\"Failed to load {f}: {e}\")
    return series

all_series = load_all_series(DATA_DIR)
if len(all_series) == 0:
    raise RuntimeError(f\"No CSV files found in {DATA_DIR}. Place your HVAC CSVs there.\")
print(f\"Loaded {len(all_series)} series. Example lengths: {[s.shape[0] for s in all_series][:8]}\")

# -------------------- Model factory --------------------
def create_model(seq_len=SEQ_LEN, feature_dim=FEATURE_DIM, hidden_dim=HIDDEN_DIM):
    inp = layers.Input(shape=(seq_len, feature_dim))
    # Use LSTM unrolled to be friendly for TFLite later if needed
    x = layers.LSTM(hidden_dim, unroll=True)(inp)
    # Predictor: predict next-step continuous channels (temp, humid, light)
    out = layers.Dense(feature_dim)(x)
    return models.Model(inp, out)

# -------------------- Task sampler --------------------
def sample_task_from_series(series, seq_len=SEQ_LEN):
    T = series.shape[0]
    if T <= seq_len:
        return None
    start = np.random.randint(0, T - seq_len - 1)
    X = series[start:start + seq_len]          # shape (seq_len, D)
    Y = series[start + seq_len]                # next-step target (D,)
    return X.astype(np.float32), Y.astype(np.float32)

def build_meta_batch(all_series, num_tasks=TASKS_PER_META):
    tasks = []
    tries = 0
    while len(tasks) < num_tasks and tries < num_tasks * 10:
        s = random.choice(all_series)
        t = sample_task_from_series(s)
        if t is not None:
            tasks.append(t)
        tries += 1
    return tasks

# -------------------- Inner update (on-task) --------------------
loss_obj = tf.keras.losses.MeanSquaredError()

@tf.function
def inner_step(model, X, Y, lr=INNER_LR):
    # Single gradient step on (X,Y). Returns loss and applies in-place update.
    with tf.GradientTape() as tape:
        pred = model(tf.expand_dims(X, 0), training=True)[0]
        loss = loss_obj(Y, pred)
    grads = tape.gradient(loss, model.trainable_variables)
    # apply gradient descent manually to model weights
    for v, g in zip(model.trainable_variables, grads):
        v.assign_sub(lr * g)
    return loss

# -------------------- Reptile meta-update --------------------
def reptile_meta_step(meta_model, tasks, inner_steps=INNER_STEPS, meta_lr=META_LR):
    init_weights = meta_model.get_weights()
    adapted_weights = []

    # For each task, clone model, do inner updates, collect weights
    for (X, Y) in tasks:
        # Create a fresh model with same architecture
        temp_model = create_model()
        temp_model.set_weights(init_weights)
        # inner loop (several gradient steps)
        for _ in range(inner_steps):
            inner_step(temp_model, X, Y, lr=INNER_LR)
        adapted_weights.append(temp_model.get_weights())

    # average adapted weights across tasks
    avg_weights = []
    for layer_weights in zip(*adapted_weights):
        avg_weights.append(np.mean(layer_weights, axis=0))

    # meta-update: move init_weights toward avg_weights by meta_lr
    new_weights = []
    for w_init, w_avg in zip(init_weights, avg_weights):
        new_weights.append(w_init + meta_lr * (w_avg - w_init))

    meta_model.set_weights(new_weights)
    return

# -------------------- Training loop --------------------
meta_model = create_model()
print(meta_model.summary())

for epoch in range(1, META_EPOCHS + 1):
    tasks = build_meta_batch(all_series, TASKS_PER_META)
    reptile_meta_step(meta_model, tasks, inner_steps=INNER_STEPS, meta_lr=META_LR)

    # quick evaluation on held-out tasks (compute MSE before/after inner adaptation)
    eval_tasks = build_meta_batch(all_series, BATCH_EVAL)
    pre_losses, post_losses = [], []
    for X, Y in eval_tasks:
        # compute pre-adaptation loss
        pred_pre = meta_model.predict(tf.expand_dims(X, 0))[0]
        pre_losses.append(np.mean((pred_pre - Y) ** 2))

        # adapt model copy
        temp_model = create_model()
        temp_model.set_weights(meta_model.get_weights())
        for _ in range(INNER_STEPS):
            inner_step(temp_model, X, Y, lr=INNER_LR)
        pred_post = temp_model.predict(tf.expand_dims(X, 0))[0]
        post_losses.append(np.mean((pred_post - Y) ** 2))

    print(f\"[Meta] Epoch {epoch}/{META_EPOCHS}  pre_loss={np.mean(pre_losses):.5f}  post_loss={np.mean(post_losses):.5f}\")

# -------------------- Save encoder weights --------------------
out_dir = \"./saved_models\"\nos.makedirs(out_dir, exist_ok=True)\nmeta_model.save(os.path.join(out_dir, \"reptile_meta_predictor.keras\"))\nprint(\"Saved meta model to\", out_dir)\n