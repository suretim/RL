#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fixed-window LSTM Meta-learning + EWC + Replay pipeline with HVAC features
- CSV columns: temp, humid, light, ac, heater, dehum, hum, label
- Sliding windows -> contrastive + FOMAML + EWC
- Export to .h5 + TFLite
"""

import os, glob, numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from global_hyparm import *
# ---------------- Hyperparameters ----------------

STEP_SIZE = 1

BATCH_SIZE = 32
EPOCHS_CONTRASTIVE = 10
EPOCHS_META = 20
INNER_LR = 1e-2
META_LR = 1e-3
REPLAY_CAPACITY = 1000
REPLAY_WEIGHT = 0.3
LAMBDA_EWC = 0.4
CONT_IDX = [0,1,2]  # temp, humid, light
HVAC_IDX = [3,4,5,6] # ac, heater, dehum, hum

# ---------------- Replay Buffer ----------------
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
    def sample(self, batch_size):
        batch_size = min(batch_size, len(self.buffer))
        idxs = np.random.choice(len(self.buffer), batch_size, replace=False)
        X_s, y_s = zip(*[self.buffer[i] for i in idxs])
        return np.array(X_s), np.array(y_s)

# ---------------- Contrastive Loss ----------------
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

# ---------------- Meta Model ----------------
class MetaModel:
    def __init__(self, num_classes=3):
        self.num_classes = num_classes
        self.encoder = self.build_lstm_encoder()
        self.model = self.build_meta_model(self.encoder)
        self.fisher = None
        self.old_params = None

    def build_lstm_encoder(self):
        inp = layers.Input(shape=(SEQ_LEN, NUM_FEATURES))
        x = layers.LSTM(FEATURE_DIM, return_sequences=True, unroll=True)(inp)
        x = layers.LSTM(FEATURE_DIM, unroll=True)(x)
        out = layers.Dense(FEATURE_DIM, activation='relu')(x)
        return models.Model(inp, out, name="lstm_encoder")

    def build_meta_model(self, encoder):
        inp = layers.Input(shape=(SEQ_LEN, NUM_FEATURES))
        z_enc = encoder(inp)
        # HVAC features
        hvac = inp[:,:,3:7]
        hvac_mean = tf.reduce_mean(hvac, axis=1)
        hvac_diff = tf.abs(hvac[:,1:,:] - hvac[:,:-1,:])
        hvac_toggle = tf.reduce_mean(hvac_diff, axis=1)
        hvac_feat = layers.Concatenate()([hvac_mean, hvac_toggle])
        hvac_feat = layers.Dense(16, activation='relu', name="hvac_dense")(hvac_feat)
        x = layers.Concatenate()([z_enc, hvac_feat])
        x = layers.Dense(64, activation='relu', name="meta_dense_64")(x)
        x = layers.Dense(32, activation='relu', name="meta_dense_32")(x)
        out = layers.Dense(self.num_classes, activation='softmax', name="meta_out")(x)
        return models.Model(inp, out, name="meta_lstm_classifier")

    # -------- Contrastive Pairs --------
    @staticmethod
    def augment_window(x):
        x_aug = x.copy()
        x_aug[:, CONT_IDX] += np.random.normal(0,0.01,x[:, CONT_IDX].shape).astype(np.float32)
        return x_aug

    @classmethod
    def make_contrastive_pairs(cls, X: np.ndarray, seq_len: int = SEQ_LEN, step_size: int = 1):
        """
        生成对比学习对 (anchor, positive)。
        安全版：跳过长度不足 seq_len 的序列
        """
        anchors, positives = [], []
        skipped = 0
        for i, x_seq in enumerate(X):
            x_seq = np.array(x_seq, dtype=np.float32)
            if len(x_seq) < seq_len:
                skipped += 1
                print(f"⚠️ 序列 {i} 長度 {len(x_seq)} 小於 SEQ_LEN={seq_len}, 已跳過")
                continue
            # 滑动窗口
            for start in range(0, len(x_seq) - seq_len + 1, step_size):
                w = x_seq[start:start + seq_len]
                anchors.append(w)
                positives.append(cls.augment_window(w))
        if len(anchors) == 0:
            raise ValueError("❌ 所有序列都太短，无法生成 contrastive pairs，请检查 CSV 或调小 SEQ_LEN")
        print(f"✅ 成功生成 {len(anchors)} contrastive pairs, 跳过 {skipped} 个短序列")
        return np.stack(anchors).astype(np.float32), np.stack(positives).astype(np.float32)

    # -------- Encode per timestep --------
    def encode_per_timestep(self, X):
        s_latent, starts = [], []
        for x_seq in X:
            for start in range(0, len(x_seq)-SEQ_LEN+1, STEP_SIZE):
                window = np.expand_dims(x_seq[start:start+SEQ_LEN],0).astype(np.float32)
                z = self.encoder(window)
                s_latent.append(z.numpy()[0])
                starts.append(start)
        return np.stack(s_latent), starts

    # -------- Save TFLite --------
    @staticmethod
    def save_tflite(model, out_path):
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
        tflite_model = converter.convert()
        with open(out_path,"wb") as f: f.write(tflite_model)
        print("Saved TFLite:", out_path)

# ---------------- Example Training Pipeline ----------------
def train_pipeline(csv_dir, tflite_out="meta_model.tflite"):
    files = glob.glob(os.path.join(csv_dir,"*.csv"))
    # Load CSVs
    X_all, y_all = [], []
    for f in files:
        df = pd.read_csv(f)
        X_all.append(df.iloc[:,:7].values)
        y_all.append(df['label'].values)
    # Flatten
    X_all = [x.astype(np.float32) for x in X_all]
    y_all = [y.astype(np.int32) for y in y_all]

    meta_model = MetaModel(num_classes=3)
    optimizer = tf.keras.optimizers.Adam(META_LR)

    # -------- Contrastive Pretrain --------
    anchors, positives = MetaModel.make_contrastive_pairs(X_all)
    dataset = tf.data.Dataset.from_tensor_slices((anchors, positives)).shuffle(2048).batch(BATCH_SIZE)
    ntxent = NTXentLoss()
    for ep in range(EPOCHS_CONTRASTIVE):
        for a,p in dataset:
            with tf.GradientTape() as tape:
                za = meta_model.encoder(a, training=True)
                zp = meta_model.encoder(p, training=True)
                loss = ntxent(za,zp)
            grads = tape.gradient(loss, meta_model.encoder.trainable_variables)
            grads = [g if g is not None else tf.zeros_like(v) for g,v in zip(grads,meta_model.encoder.trainable_variables)]
            optimizer.apply_gradients(zip(grads, meta_model.encoder.trainable_variables))
        print(f"[Contrastive] Epoch {ep+1}/{EPOCHS_CONTRASTIVE}, loss={float(loss.numpy()):.4f}")

    # -------- 可以加 FOMAML / Replay / EWC --------
    # 這裡省略，可套用你現有 outer_update_with_lll 函數

    # -------- Save TFLite / h5 --------
    meta_model.model.save("meta_model.h5")
    MetaModel.save_tflite(meta_model.model, tflite_out)

    print("✅ Pipeline 完成")

# ---------------- Main ----------------
if __name__ == "__main__":
    train_pipeline(DATA_DIR, tflite_out="meta_model.tflite")
