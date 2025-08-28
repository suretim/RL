#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, glob
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
from collections import deque
import random

# =============================
# 超参数
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

np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

# =============================
# 1) 加载 CSV -> 滑窗样本
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
    X_labeled = np.empty((0, SEQ_LEN, X_unlabeled.shape[2] if X_unlabeled.size > 0 else 3), dtype=np.float32)
    y_labeled = np.empty((0,), dtype=np.int32)

NUM_FEATS = X_labeled.shape[2] if X_labeled.size > 0 else (X_unlabeled.shape[2] if X_unlabeled.size > 0 else 3)

# =============================
# 2) 对比学习
# =============================
def augment_window(x):
    return x + np.random.normal(0, 0.01, x.shape).astype(np.float32)

def make_contrastive_pairs(X):
    anchors, positives = [], []
    for w in X:
        anchors.append(w)
        positives.append(augment_window(w))
    return np.stack(anchors).astype(np.float32), np.stack(positives).astype(np.float32)

class NTXentLoss(tf.keras.losses.Loss):
    def __init__(self, temperature=0.1):
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

def build_lstm_encoder(seq_len, num_feats, feature_dim=FEATURE_DIM):
    inp = layers.Input(shape=(seq_len, num_feats))
    x = layers.LSTM(feature_dim)(inp)
    out = layers.Dense(feature_dim, activation="relu")(x)
    return models.Model(inp, out, name="lstm_encoder")

lstm_encoder = build_lstm_encoder(SEQ_LEN, NUM_FEATS, FEATURE_DIM)
contrastive_opt = tf.keras.optimizers.Adam()
ntxent = NTXentLoss(temperature=0.2)

if X_unlabeled.size == 0:
    X_unlabeled = np.random.randn(200, SEQ_LEN, NUM_FEATS).astype(np.float32)

anchors, positives = make_contrastive_pairs(X_unlabeled)
contrast_ds = tf.data.Dataset.from_tensor_slices((anchors, positives)).shuffle(2048).batch(BATCH_SIZE)

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

# =============================
# 3) FOMAML + LLL + EWC
# =============================
def build_meta_model(encoder, num_classes=NUM_CLASSES):
    inp = layers.Input(shape=(SEQ_LEN, NUM_FEATS))
    x = encoder(inp)
    x = layers.Dense(32, activation="relu")(x)
    out = layers.Dense(num_classes, activation="softmax")(x)
    return models.Model(inp, out, name="meta_lstm_classifier")

meta_model = build_meta_model(lstm_encoder, NUM_CLASSES)
meta_optimizer = tf.keras.optimizers.Adam(META_LR)

def sample_tasks(X, y, num_tasks=NUM_TASKS, support_size=SUPPORT_SIZE, query_size=QUERY_SIZE):
    tasks = []
    n = len(X)
    if n < support_size + query_size:
        raise ValueError(f"样本不足以构建任务：需要 {support_size+query_size}，但只有 {n}")
    for _ in range(num_tasks):
        idx = np.random.choice(n, support_size + query_size, replace=False)
        X_support, y_support = X[idx[:support_size]], y[idx[:support_size]]
        X_query, y_query = X[idx[support_size:]], y[idx[support_size:]]
        tasks.append((X_support, y_support, X_query, y_query))
    return tasks

# ---------- inner update ----------
def inner_update(model, X_support, y_support, lr_inner=INNER_LR):
    with tf.GradientTape() as tape:
        preds_support = model(X_support, training=True)
        loss_support = tf.reduce_mean(
            tf.keras.losses.sparse_categorical_crossentropy(y_support, preds_support)
        )
    grads_inner = tape.gradient(loss_support, model.trainable_variables)
    updated_vars = [w - lr_inner * g for w, g in zip(model.trainable_variables, grads_inner)]
    return updated_vars

# ---------- Replay Buffer with Reservoir Sampling ----------
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
                # reservoir sampling
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

# ---------- outer update with LLL + EWC ----------
def outer_update_with_lll(meta_model, meta_optimizer, tasks,
                          lr_inner=INNER_LR, replay_weight=REPLAY_WEIGHT,
                          lambda_ewc=LAMBDA_EWC, prev_weights=None):
    meta_grads = [tf.zeros_like(v) for v in meta_model.trainable_variables]
    query_acc_list, query_loss_list = [], []

    for X_support, y_support, X_query, y_query in tasks:
        # 保存原始权重
        orig_vars = [tf.identity(v) for v in meta_model.trainable_variables]

        # inner update
        updated_vars = inner_update(meta_model, X_support, y_support)
        for var, upd in zip(meta_model.trainable_variables, updated_vars):
            var.assign(upd)

        # query loss + replay + EWC
        with tf.GradientTape() as tape:
            preds_q = meta_model(X_query, training=True)
            loss_q = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(y_query, preds_q))
            loss_total = loss_q

            # replay regularization
            if len(memory) >= 8:
                X_old, y_old = memory.sample(batch_size=32)
                preds_old = meta_model(X_old, training=True)
                replay_loss = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(y_old, preds_old))
                loss_total = (1 - replay_weight) * loss_total + replay_weight * replay_loss

            # EWC regularization
            if prev_weights is not None:
                ewc_loss = 0.0
                for w, pw in zip(meta_model.trainable_variables, prev_weights):
                    ewc_loss += tf.reduce_sum(tf.square(w - pw))
                loss_total += lambda_ewc * ewc_loss

        grads = tape.gradient(loss_total, meta_model.trainable_variables)
        meta_grads = [mg + g / len(tasks) for mg, g in zip(meta_grads, grads)]

        # accuracy
        q_acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(preds_q, axis=1), y_query), tf.float32))
        query_acc_list.append(float(q_acc.numpy()))
        query_loss_list.append(float(loss_q.numpy()))

        # 恢复原始权重
        for var, orig in zip(meta_model.trainable_variables, orig_vars):
            var.assign(orig)

        # 更新 replay buffer
        memory.add(X_support, y_support)
        memory.add(X_query, y_query)

    # meta update
    meta_optimizer.apply_gradients(zip(meta_grads, meta_model.trainable_variables))

    return float(np.mean(query_loss_list)), float(np.mean(query_acc_list)), [tf.identity(v) for v in meta_model.trainable_variables]

# ======= 训练元学习 =======
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
    print("跳过元学习：没有有标签数据。")

# =============================
# 4) 可视化
# =============================
plt.figure()
plt.plot(contrastive_loss_history, label="Contrastive Loss")
plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("Contrastive Learning")
plt.legend(); plt.grid(True); plt.show()

if meta_loss_history:
    plt.figure()
    plt.plot(meta_loss_history, label="Query Loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("FOMAML + LLL + EWC Loss")
    plt.legend(); plt.grid(True); plt.show()

    plt.figure()
    plt.plot(meta_acc_history, label="Query Accuracy")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.title("FOMAML + LLL + EWC Accuracy")
    plt.legend(); plt.grid(True); plt.show()

# =============================
# 5) TFLite 导出
# =============================
def save_tflite(model, out_path):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
    tflite_model = converter.convert()
    with open(out_path, "wb") as f:
        f.write(tflite_model)
    print("Saved TFLite:", out_path)

save_tflite(lstm_encoder, "lstm_encoder_contrastive.tflite")
if X_labeled.size > 0:
    save_tflite(meta_model, "meta_lstm_classifier.tflite")
