#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Meta-learning pipeline with HVAC-aware features and flowering-period focus.
- Expects CSV columns: temp, humid, light, ac, heater, dehum, hum, label
- Sliding windows -> contrastive learning (unlabeled) + FOMAML with LLL + EWC (labeled)
- Encoder: LSTM on continuous features only (temp/humid/light)
- Additional HVAC features: mean on/off rate + toggle rate (abs(diff)) over time
- Gradient boost on flowering period with abnormal HVAC toggling
- TFLite export restricted to TFLITE_BUILTINS
"""

import os, glob
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
import random
import matplotlib.pyplot as plt

# =============================
# Hyperparameters
# =============================
DATA_GLOB = "./data/*.csv"
SEQ_LEN = 64
FEATURE_DIM = 64
BATCH_SIZE = 32
EPOCHS_CONTRASTIVE = 10
EPOCHS_META = 20
INNER_LR = 1e-2
META_LR = 1e-3
NUM_CLASSES = 3
NUM_TASKS = 5
SUPPORT_SIZE = 10
QUERY_SIZE = 20
REPLAY_CAPACITY = 1000
REPLAY_WEIGHT = 0.3
LAMBDA_EWC = 1e-3
FLOWERING_WEIGHT = 2.0  # gradient boost upper bound for flowering-focus

np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

# =============================
# 1) Load CSV -> Sliding windows
# =============================
X_labeled_list, y_labeled_list = [], []
X_unlabeled_list = []

for file in sorted(glob.glob(DATA_GLOB)):
    df = pd.read_csv(file).fillna(-1)
    data = df.values.astype(np.float32)
    feats, labels = data[:, :-1], data[:, -1]
    for i in range(len(data) - SEQ_LEN + 1):
        w_x = feats[i:i + SEQ_LEN]
        w_y = labels[i + SEQ_LEN - 1]
        if w_y == -1:
            X_unlabeled_list.append(w_x)
        else:
            X_labeled_list.append(w_x)
            y_labeled_list.append(int(w_y))

X_unlabeled = np.array(X_unlabeled_list, dtype=np.float32) if len(X_unlabeled_list) > 0 else np.empty((0,), dtype=np.float32)
if len(X_labeled_list) > 0:
    X_labeled = np.array(X_labeled_list, dtype=np.float32)
    y_labeled = np.array(y_labeled_list, dtype=np.int32)
else:
    # fallback shape (not ideal; expect at least unlabeled data present)
    X_labeled = np.empty((0, SEQ_LEN, X_unlabeled.shape[2] if X_unlabeled.size > 0 else 7), dtype=np.float32)
    y_labeled = np.empty((0,), dtype=np.int32)

NUM_FEATS = X_labeled.shape[2] if X_labeled.size > 0 else (X_unlabeled.shape[2] if X_unlabeled.size > 0 else 7)

# Verify hvac columns exist
if NUM_FEATS < 7:
    raise ValueError("Expected at least 7 features per timestep: [temp, humid, light, ac, heater, dehum, hum]. Found: %d" % NUM_FEATS)

# Index conventions
CONT_IDX = [0, 1, 2]   # temp, humid, light
HVAC_IDX = [3, 4, 5, 6]  # ac, heater, dehum, hum

# =============================
# 2) Contrastive learning
# =============================
def augment_window(x):
    """Only perturb continuous channels to keep binary HVAC channels intact."""
    x_aug = x.copy()
    x_aug[:, CONT_IDX] = x[:, CONT_IDX] + np.random.normal(0, 0.01, x[:, CONT_IDX].shape).astype(np.float32)
    return x_aug

def make_contrastive_pairs(X):
    anchors, positives = [], []
    for w in X:
        anchors.append(w)
        positives.append(augment_window(w))
    return np.stack(anchors).astype(np.float32), np.stack(positives).astype(np.float32)

class NTXentLoss(tf.keras.losses.Loss):
    def __init__(self, temperature=0.2):
        super().__init__()
        self.temperature = temperature
    def call(self, z_i, z_j):
        z_i = tf.math.l2_normalize(z_i, axis=1)
        z_j = tf.math.l2_normalize(z_j, axis=1)
        logits = tf.matmul(z_i, z_j, transpose_b=True) / self.temperature
        labels = tf.range(tf.shape(z_i)[0])
        loss_i = tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)
        loss_j = tf.keras.losses.sparse_categorical_crossentropy(labels, tf.transpose(logits), from_logits=True)
        return tf.reduce_mean(loss_i + loss_j)

# =============================
# 2) LSTM Encoder (unroll=True, continuous only)
# =============================
def build_lstm_encoder(seq_len, num_feats, feature_dim=FEATURE_DIM):
    inp = layers.Input(shape=(seq_len, num_feats))
    x_cont = layers.Lambda(lambda z: z[:, :, :3])(inp)  # [B,T,3]
    x = layers.LSTM(feature_dim, unroll=True)(x_cont)
    out = layers.Dense(feature_dim, activation="relu")(x)
    return models.Model(inp, out, name="lstm_encoder")

# =============================
# 3) Meta Model (Encoder + HVAC features)
# =============================
def build_meta_model(encoder, num_classes=NUM_CLASSES):
    inp = layers.Input(shape=(SEQ_LEN, NUM_FEATS))   # expect >=7
    z_enc = encoder(inp)  # [B, FEATURE_DIM]

    # HVAC slice
    hvac = layers.Lambda(lambda z: z[:, :, 3:7])(inp)   # [B,T,4]
    hvac_mean = layers.Lambda(lambda z: tf.reduce_mean(z, axis=1))(hvac)  # [B,4]

    # Toggle rate via abs(diff)
    hvac_shift = layers.Lambda(lambda z: z[:, 1:, :])(hvac)      # [B,T-1,4]
    hvac_prev  = layers.Lambda(lambda z: z[:, :-1, :])(hvac)     # [B,T-1,4]
    hvac_diff  = layers.Lambda(lambda t: tf.abs(t[0] - t[1]))([hvac_shift, hvac_prev])  # [B,T-1,4]
    hvac_toggle_rate = layers.Lambda(lambda z: tf.reduce_mean(z, axis=1))(hvac_diff)    # [B,4]

    hvac_feat = layers.Concatenate()([hvac_mean, hvac_toggle_rate])  # [B,8]
    hvac_feat = layers.Dense(16, activation="relu")(hvac_feat)

    x = layers.Concatenate()([z_enc, hvac_feat])
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dense(32, activation="relu")(x)
    out = layers.Dense(NUM_CLASSES, activation="softmax")(x)
    return models.Model(inp, out, name="meta_lstm_classifier")

# Build models
lstm_encoder = build_lstm_encoder(SEQ_LEN, NUM_FEATS, FEATURE_DIM)
contrastive_opt = tf.keras.optimizers.Adam()
ntxent = NTXentLoss(temperature=0.2)

# Provide unlabeled fallback if none
if X_unlabeled.size == 0:
    X_unlabeled = np.random.randn(200, SEQ_LEN, NUM_FEATS).astype(np.float32)

anchors, positives = make_contrastive_pairs(X_unlabeled)
contrast_ds = tf.data.Dataset.from_tensor_slices((anchors, positives)).shuffle(2048).batch(BATCH_SIZE)

# Train contrastive
contrastive_loss_history = []
for ep in range(EPOCHS_CONTRASTIVE):
    for a, p in contrast_ds:
        with tf.GradientTape() as tape:
            za = lstm_encoder(a, training=True)
            zp = lstm_encoder(p, training=True)
            loss = ntxent(za, zp)
        grads = tape.gradient(loss, lstm_encoder.trainable_variables)
        contrastive_opt.apply_gradients(zip(grads, lstm_encoder.trainable_variables))
    contrastive_loss_history.append(float(loss.numpy()))
    print(f"[Contrastive] Epoch {ep+1}/{EPOCHS_CONTRASTIVE}, loss={float(loss.numpy()):.4f}")

# Meta model
meta_model = build_meta_model(lstm_encoder, NUM_CLASSES)
meta_optimizer = tf.keras.optimizers.Adam(META_LR)

def sample_tasks(X, y, num_tasks=NUM_TASKS, support_size=SUPPORT_SIZE, query_size=QUERY_SIZE):
    tasks = []
    n = len(X)
    if n < support_size + query_size:
        raise ValueError(f"Not enough labeled samples to build tasks: need {support_size+query_size}, got {n}")
    for _ in range(num_tasks):
        idx = np.random.choice(n, support_size + query_size, replace=False)
        X_support, y_support = X[idx[:support_size]], y[idx[:support_size]]
        X_query, y_query = X[idx[support_size:]], y[idx[support_size:]]
        tasks.append((X_support, y_support, X_query, y_query))
    return tasks

def inner_update(model, X_support, y_support, lr_inner=INNER_LR):
    with tf.GradientTape() as tape:
        preds_support = model(X_support, training=True)
        loss_support = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(y_support, preds_support))
    grads_inner = tape.gradient(loss_support, model.trainable_variables)
    updated_vars = [w - lr_inner * g for w, g in zip(model.trainable_variables, grads_inner)]
    return updated_vars

class ReplayBuffer:
    def __init__(self, capacity=REPLAY_CAPACITY):
        self.buffer = []
        self.capacity = capacity
        self.n_seen = 0
    def add(self, X, y):
        for xi, yi in zip(X, y):
            self.n_seen += 1
            if len(self.buffer) < self.capacity:
                self.buffer.append((xi, yi))
            else:
                r = np.random.randint(0, self.n_seen)
                if r < self.capacity:
                    self.buffer[r] = (xi, yi)
    def __len__(self):
        return len(self.buffer)
    def sample(self, batch_size):
        batch_size = min(batch_size, len(self.buffer))
        idxs = np.random.choice(len(self.buffer), batch_size, replace=False)
        X_s, y_s = zip(*[self.buffer[i] for i in idxs])
        return np.array(X_s), np.array(y_s)

memory = ReplayBuffer(capacity=REPLAY_CAPACITY)

# ===== Helpers for flowering focus =====
def is_flowering_seq(x_seq, light_idx=2, th_light=550.0):
    light_mean = float(np.mean(x_seq[:, light_idx]))
    return light_mean >= th_light

def hvac_toggle_score(x_seq, hvac_slice=slice(3,7), th_toggle=0.15):
    hv = x_seq[:, hvac_slice]  # [T,4]
    if hv.shape[0] < 2:
        return 0.0, False
    diff = np.abs(hv[1:] - hv[:-1])   # [T-1,4]
    rate = float(diff.mean())
    return rate, rate >= th_toggle

def outer_update_with_lll(meta_model, meta_optimizer, tasks,
                          lr_inner=INNER_LR, replay_weight=REPLAY_WEIGHT,
                          lambda_ewc=LAMBDA_EWC, prev_weights=None):
    meta_grads = [tf.zeros_like(v) for v in meta_model.trainable_variables]
    query_acc_list, query_loss_list = [], []

    for X_support, y_support, X_query, y_query in tasks:
        orig_vars = [tf.identity(v) for v in meta_model.trainable_variables]

        # inner update
        updated_vars = inner_update(meta_model, X_support, y_support)
        for var, upd in zip(meta_model.trainable_variables, updated_vars):
            var.assign(upd)

        with tf.GradientTape() as tape:
            preds_q = meta_model(X_query, training=True)
            loss_q = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(y_query, preds_q))
            loss_total = loss_q

            # replay
            if len(memory) >= 8:
                X_old, y_old = memory.sample(batch_size=32)
                preds_old = meta_model(X_old, training=True)
                replay_loss = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(y_old, preds_old))
                loss_total = (1 - replay_weight) * loss_total + replay_weight * replay_loss

            # EWC (L2 to prev weights as proxy)
            if prev_weights is not None:
                ewc_loss = 0.0
                for w, pw in zip(meta_model.trainable_variables, prev_weights):
                    ewc_loss += tf.reduce_sum(tf.square(w - pw))
                loss_total += lambda_ewc * ewc_loss

        grads = tape.gradient(loss_total, meta_model.trainable_variables)

        # ===== Flowering + HVAC toggling gradient boost =====
        flowering_mask = []
        toggle_scores = []
        for i in range(len(X_query)):
            x_seq = X_query[i]  # [T,D]
            flw = is_flowering_seq(x_seq, light_idx=2, th_light=550.0)
            tscore, tabove = hvac_toggle_score(x_seq, hvac_slice=slice(3,7), th_toggle=0.15)
            flowering_mask.append(bool(flw and tabove))
            toggle_scores.append(tscore)

        if any(flowering_mask):
            ratio = sum(flowering_mask) / len(flowering_mask)
            mean_toggle = np.mean([t for m,t in zip(flowering_mask, toggle_scores) if m]) if any(flowering_mask) else 0.0
            toggle_boost = min(1.0 + float(mean_toggle)*2.0, FLOWERING_WEIGHT)
            boost = 1.0 + (FLOWERING_WEIGHT - 1.0) * ratio
            total_boost = float(min(boost * toggle_boost, FLOWERING_WEIGHT))
            grads = [g * total_boost for g in grads]

        meta_grads = [mg + g / len(tasks) for mg, g in zip(meta_grads, grads)]

        q_acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(preds_q, axis=1), y_query), tf.float32))
        query_acc_list.append(float(q_acc.numpy()))
        query_loss_list.append(float(loss_q.numpy()))

        # restore original vars
        for var, orig in zip(meta_model.trainable_variables, orig_vars):
            var.assign(orig)

        # update memory
        memory.add(X_support, y_support)
        memory.add(X_query, y_query)

    meta_optimizer.apply_gradients(zip(meta_grads, meta_model.trainable_variables))
    return float(np.mean(query_loss_list)), float(np.mean(query_acc_list)), [tf.identity(v) for v in meta_model.trainable_variables]

# ======= Train meta-learning =======
meta_loss_history, meta_acc_history = [], []
prev_weights = None

if X_labeled.size > 0:
    for ep in range(EPOCHS_META):
        tasks = sample_tasks(X_labeled, y_labeled)
        loss, acc, prev_weights = outer_update_with_lll(meta_model, meta_optimizer, tasks, prev_weights=prev_weights)
        meta_loss_history.append(loss)
        meta_acc_history.append(acc)
        print(f"[Meta] Epoch {ep+1}/{EPOCHS_META}, loss={loss:.4f}, acc={acc:.4f}")
else:
    print("Skip meta-learning: no labeled data.")

# =============================
# 4) Visualization
# =============================
plt.figure()
plt.plot(contrastive_loss_history, label="Contrastive Loss")
plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("Contrastive Learning")
plt.legend(); plt.grid(True); plt.tight_layout()

if meta_loss_history:
    plt.figure()
    plt.plot(meta_loss_history, label="Query Loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("FOMAML + LLL + EWC Loss")
    plt.legend(); plt.grid(True); plt.tight_layout()

    plt.figure()
    plt.plot(meta_acc_history, label="Query Accuracy")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.title("FOMAML + LLL + EWC Accuracy")
    plt.legend(); plt.grid(True); plt.tight_layout()

# =============================
# 5) TFLite export (BUILTINS only)
# =============================
def save_tflite(model, out_path):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    tflite_model = converter.convert()
    with open(out_path, "wb") as f:
        f.write(tflite_model)
    print("Saved TFLite:", out_path)

# Save models
save_tflite(lstm_encoder, "lstm_encoder_contrastive.tflite")
if X_labeled.size > 0:
    save_tflite(meta_model, "meta_lstm_classifier.tflite")

print("Done.")
