#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models

# =============================
# 超参数
# =============================
DATA_GLOB = "./data/*.csv"         # CSV: 每行 [feat1,...,featK,label]；无标签用 -1
SEQ_LEN = 64
FEATURE_DIM = 64                   # 编码器输出维度
BATCH_SIZE = 32

EPOCHS_CONTRASTIVE = 10            # 对比学习预训练轮数（无标签）
EPOCHS_META = 30                   # 元学习轮数（有标签）
INNER_STEPS = 1                    # 任务内微调步数
INNER_LR = 1e-2                    # 任务内学习率
META_LR = 1e-3                     # 元更新学习率
REPTILE_EPS = 0.1                  # Reptile 步长（meta 权重向 task 权重靠拢的比例）

NUM_CLASSES = 3

np.random.seed(42)
tf.random.set_seed(42)

# =============================
# 1) 加载 CSV -> 滑窗样本 + 记录每个窗口的 task_id
#    约定：每个 CSV 代表一个“任务域”（不同植物/环境），便于元学习抽 task
# =============================
X_labeled_list, y_labeled_list, task_id_list = [], [], []
X_unlabeled_list = []

csv_files = sorted(glob.glob(DATA_GLOB))
if len(csv_files) == 0:
    raise FileNotFoundError(f"未找到数据：{DATA_GLOB}")

for tid, file in enumerate(csv_files):
    df = pd.read_csv(file).fillna(-1)  # NaN -> -1 表示无标签
    data = df.values.astype(np.float32)
    feats = data[:, :-1]
    labels = data[:, -1]

    # 构建滑窗
    for i in range(len(data) - SEQ_LEN + 1):
        window_x = feats[i:i+SEQ_LEN]
        window_y = labels[i+SEQ_LEN-1]  # 用窗口末尾的标签
        if window_y == -1:
            X_unlabeled_list.append(window_x)
        else:
            X_labeled_list.append(window_x)
            y_labeled_list.append(int(window_y))
            task_id_list.append(tid)  # 该窗口从哪个 CSV 来

X_unlabeled = np.array(X_unlabeled_list, dtype=np.float32)
if len(X_labeled_list) > 0:
    X_labeled = np.array(X_labeled_list, dtype=np.float32)
    y_labeled = np.array(y_labeled_list, dtype=np.int32)
    task_ids   = np.array(task_id_list,  dtype=np.int32)
    print("有标签样本:", X_labeled.shape, "任务域数:", len(csv_files))
else:
    X_labeled = np.empty((0, SEQ_LEN, X_unlabeled.shape[-1] if len(X_unlabeled)>0 else 3), dtype=np.float32)
    y_labeled = np.empty((0,), dtype=np.int32)
    task_ids  = np.empty((0,), dtype=np.int32)
    print("警告：没有有标签样本，后续仅能做无监督预训练和导出编码器。")

NUM_FEATS = (X_labeled.shape[2] if X_labeled.size>0 else
             (X_unlabeled.shape[2] if X_unlabeled.size>0 else 3))

# =============================
# 2) 对比学习（SimCLR 风格）预训练 LSTM 编码器
# =============================
def augment_window(x):
    # 轻微高斯噪声 + 随机时序丢帧/抖动（简化版）
    x = x + np.random.normal(0, 0.01, x.shape).astype(np.float32)
    return x

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
    x = layers.LSTM(feature_dim, return_sequences=False)(inp)
    out = layers.Dense(feature_dim, activation="relu")(x)
    return models.Model(inp, out, name="lstm_encoder")

lstm_encoder = build_lstm_encoder(SEQ_LEN, NUM_FEATS, FEATURE_DIM)
contrastive_opt = tf.keras.optimizers.Adam()

# 若无无标签数据，生成一些随机窗口做预训练
if X_unlabeled.size == 0:
    print("没有无标签数据，生成随机数据用于对比学习预训练……")
    X_unlabeled = np.random.randn(200, SEQ_LEN, NUM_FEATS).astype(np.float32)

anchors, positives = make_contrastive_pairs(X_unlabeled)
contrast_ds = tf.data.Dataset.from_tensor_slices((anchors, positives)).shuffle(2048).batch(BATCH_SIZE)

ntxent = NTXentLoss(temperature=0.2)
for ep in range(EPOCHS_CONTRASTIVE):
    for a, p in contrast_ds:
        with tf.GradientTape() as tape:
            za = lstm_encoder(a, training=True)
            zp = lstm_encoder(p, training=True)
            loss = ntxent(za, zp)
        grads = tape.gradient(loss, lstm_encoder.trainable_variables)
        contrastive_opt.apply_gradients(zip(grads, lstm_encoder.trainable_variables))
    print(f"[Contrastive] Epoch {ep+1}/{EPOCHS_CONTRASTIVE} loss={float(loss.numpy()):.4f}")

# =============================
# 3) 元学习（Reptile）
#    思想：对每个任务，复制模型 -> 在 support 上做若干步微调
#          将 meta 权重朝“任务后权重”移动（平均多个任务）
# =============================
def build_meta_model(encoder, num_classes=NUM_CLASSES):
    # 复制一个具有相同结构的编码器 + 分类头
    inp = layers.Input(shape=(SEQ_LEN, NUM_FEATS))
    x = encoder(inp)  # 使用已有编码器作为 backbone
    x = layers.Dense(32, activation="relu")(x)
    out = layers.Dense(num_classes, activation="softmax")(x)
    return models.Model(inp, out, name="meta_lstm_classifier")

meta_model = build_meta_model(lstm_encoder, NUM_CLASSES)
meta_optimizer = tf.keras.optimizers.Adam(META_LR)

# 将有标签数据按 task_id 聚合，便于采样任务
task_to_indices = {}
for idx, tid in enumerate(task_ids):
    task_to_indices.setdefault(int(tid), []).append(idx)
task_ids_unique = list(task_to_indices.keys())

def sample_task_batches(X, y, idx_list, batch_size):
    """ 从某任务的样本索引集合中，采一个 batch（用于 support/query） """
    idx = np.random.choice(idx_list, size=min(batch_size, len(idx_list)), replace=False)
    return X[idx], y[idx]

#def clone_weights(model):
#    return [w.numpy().copy() for w in model.get_weights()]
def clone_weights(model):
    # 直接 copy numpy 数组即可
    return [w.copy() for w in model.get_weights()]

def assign_weights(model, weights_list):
    model.set_weights(weights_list)

def reptile_meta_step(meta_model, task_ids_unique, inner_steps=INNER_STEPS,
                      inner_lr=INNER_LR, reptile_eps=REPTILE_EPS,
                      support_bs=32, query_bs=64):
    """
    执行一次 Reptile 外层更新：
      1) 采样若干任务
      2) 对每个任务：克隆模型 -> 在 support 上做 inner_steps 次 SGD
      3) 收集任务后权重，meta 权重朝其平均值移动
    """
    if len(task_ids_unique) == 0:
        return  # 没有有标签任务可用

    # 保存 meta 初始权重
    meta_w = clone_weights(meta_model)
    task_weights = []

    for tid in task_ids_unique:
        idx_list = task_to_indices[tid]
        # 克隆一个任务模型（结构相同，复制 meta 当前权重）
        task_model = build_meta_model(lstm_encoder, NUM_CLASSES)
        assign_weights(task_model, meta_w)
        # 任务内优化器（独立于 meta_optimizer）
        inner_opt = tf.keras.optimizers.SGD(learning_rate=inner_lr)

        # inner loop：在该任务的 support 上做若干步
        for _ in range(inner_steps):
            Xs, ys = sample_task_batches(X_labeled, y_labeled, idx_list, support_bs)
            with tf.GradientTape() as tape:
                preds = task_model(Xs, training=True)
                loss = tf.keras.losses.sparse_categorical_crossentropy(ys, preds)
                loss = tf.reduce_mean(loss)
            grads = tape.gradient(loss, task_model.trainable_variables)
            inner_opt.apply_gradients(zip(grads, task_model.trainable_variables))

        # （可选）在 query 上评估一下
        Xq, yq = sample_task_batches(X_labeled, y_labeled, idx_list, query_bs)
        q_preds = task_model(Xq, training=False)
        q_acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(q_preds, axis=1), yq), tf.float32))
        print(f"  Task {tid}: query_acc={float(q_acc.numpy()):.3f}")

        # 收集该任务训练后的权重
        task_weights.append(clone_weights(task_model))

    # 将 meta 权重朝各任务权重的平均值靠拢（Reptile 更新）
    avg_task_w = []
    for weights_per_layer in zip(*task_weights):
        avg_task_w.append(np.mean(np.stack(weights_per_layer, axis=0), axis=0))

    # new_meta = meta + eps * (avg_task - meta)
    new_meta_w = [mw + reptile_eps * (aw - mw) for mw, aw in zip(meta_w, avg_task_w)]
    assign_weights(meta_model, new_meta_w)

# ======= 元训练循环 =======
if X_labeled.size > 0:
    for ep in range(EPOCHS_META):
        print(f"[Meta] Epoch {ep+1}/{EPOCHS_META}")
        reptile_meta_step(meta_model, task_ids_unique,
                          inner_steps=INNER_STEPS, inner_lr=INNER_LR,
                          reptile_eps=REPTILE_EPS, support_bs=BATCH_SIZE, query_bs=BATCH_SIZE*2)
else:
    print("跳过元学习：没有有标签数据。")

# =============================
# 4) 可选：在整体验证集上做一个快速评估（把所有有标签样本打乱切分）
# =============================
if X_labeled.size > 0:
    perm = np.random.permutation(len(X_labeled))
    split = int(0.8 * len(perm))
    tr_idx, va_idx = perm[:split], perm[split:]
    Xtr, ytr = X_labeled[tr_idx], y_labeled[tr_idx]
    Xva, yva = X_labeled[va_idx], y_labeled[va_idx]

    # 仅微调最后分类头（模拟 few-shot 适应场景）
    meta_model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                       loss="sparse_categorical_crossentropy",
                       metrics=["accuracy"])
    hist = meta_model.fit(Xtr, ytr, validation_data=(Xva, yva),
                          epochs=5, batch_size=BATCH_SIZE, verbose=1)
    print("Val acc (after short finetune):", hist.history["val_accuracy"][-1])

# =============================
# 5) 导出 TFLite：编码器 + 元分类器
# =============================
def save_tflite(model, out_path):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS
    ]
    converter._experimental_lower_tensor_list_ops = False
    tflite_model = converter.convert()
    with open(out_path, "wb") as f:
        f.write(tflite_model)
    print("Saved TFLite:", out_path)

# 独立导出编码器（用于边端只提特征）
save_tflite(lstm_encoder, "lstm_encoder_contrastive.tflite")
# 导出元学习后的端到端分类器（编码器 + 分类头）
if X_labeled.size > 0:
    save_tflite(meta_model, "meta_lstm_classifier.tflite")
