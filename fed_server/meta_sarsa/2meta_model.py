#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fixed-window LSTM Meta-learning + EWC + Replay pipeline with HVAC features
- CSV columns: temp, humid, light, ac, heater, dehum, hum, label
- Sliding windows -> contrastive + FOMAML + EWC
- Export to .h5 + TFLite
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))  # RL 根目录
from global_hyparm import *
import glob
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models

from utils_module import sample_tasks,load_csv_data
from utils_fisher import NTXentLoss, ReplayBuffer, MetaModel
import argparse
DATA_DIR  = "../../../data/sarsa"
META_OUT_TF="../meta_model.tflite"
# ---------------- Hyperparameters ----------------
ENCODER_MODE = "freeze"  # one of {"finetune","freeze","last_n"}
LAST_N = 1
STEP_SIZE = 1

BATCH_SIZE = 32
EPOCHS_CONTRASTIVE = 1 
EPOCHS_META = 1
INNER_LR = 1e-2
META_LR = 1e-3
REPLAY_CAPACITY = 1000
REPLAY_WEIGHT = 0.3
LAMBDA_EWC = 0.4
CONT_IDX = [0,1,2]  # temp, humid, light
HVAC_IDX = [3,4,5,6] # ac, heater, dehum, hum

# ---------------- Meta Model ----------------
class MetaModel_aug(MetaModel):
    def __init__(self, num_classes=3, seq_len=SEQ_LEN, num_feats=NUM_FEATURES, feature_dim=FEATURE_DIM):
        super().__init__(trainable=True,num_classes=num_classes, seq_len=seq_len, num_feats=num_feats, feature_dim=feature_dim)
        self.num_classes = num_classes
        #self.encoder = self.build_lstm_encoder()
        #self.model = self.build_meta_model(self.encoder)
        self.fisher = None
        self.old_params = None

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



# ---------------- Example Training Pipeline ----------------
def train_pipeline(csv_dir, tflite_out=META_OUT_TF):
    # ===== Load data =====
    load_glob = os.path.join(LOAD_DIR, f"*.csv")
    X_unlabeled, X_labeled, y_labeled, num_feats = load_csv_data(load_glob, SEQ_LEN)

    model = MetaModel_aug(num_classes=3)
    optimizer = tf.keras.optimizers.Adam(META_LR)

    # -------- Contrastive Pretrain --------
    anchors, positives = MetaModel_aug.make_contrastive_pairs(X_labeled)
    dataset = tf.data.Dataset.from_tensor_slices((anchors, positives)).shuffle(2048).batch(BATCH_SIZE)
    ntxent = NTXentLoss()
    for ep in range(EPOCHS_CONTRASTIVE):
        for a,p in dataset:
            with tf.GradientTape() as tape:
                za = model.encoder(a, training=True)
                zp = model.encoder(p, training=True)
                loss = ntxent(za,zp)
            grads = tape.gradient(loss, model.encoder.trainable_variables)
            grads = [g if g is not None else tf.zeros_like(v) for g,v in zip(grads,model.encoder.trainable_variables)]
            optimizer.apply_gradients(zip(grads, model.encoder.trainable_variables))
        print(f"[Contrastive] Epoch {ep+1}/{EPOCHS_CONTRASTIVE}, loss={float(loss.numpy()):.4f}")

    # -------- 可以加 FOMAML / Replay / EWC --------
    # 這裡省略，可套用你現有 outer_update_with_lll 函數
    # ===== Meta model =====
    #meta_model = model.build_meta_model(lstm_encoder)
    meta_encoder=model.model
    lstm_encoder=model.encoder
    meta_optimizer = tf.keras.optimizers.Adam(META_LR)
    model.set_trainable_layers(lstm_encoder, meta_encoder, ENCODER_MODE, LAST_N)

    memory = ReplayBuffer(capacity=REPLAY_CAPACITY)
    prev_weights = None
    fisher_matrix = None

    if X_labeled.size > 0:
        fisher_matrix = model.compute_fisher_matrix(meta_encoder, X_labeled, y_labeled)
        for ep in range(EPOCHS_META):
            tasks = sample_tasks(X_labeled, y_labeled,num_tasks=5)
            loss, acc, prev_weights = model.outer_update_with_lll(
                memory=memory,
                meta_model=meta_encoder,
                meta_optimizer=meta_optimizer,
                tasks=tasks,
                lr_inner=INNER_LR,
                prev_weights=prev_weights,
                fisher_matrix=fisher_matrix,
            )
            print(f"[Meta ] Epoch {ep + 1}/{EPOCHS_META}, loss={loss:.4f}, acc={acc:.4f}")
    else:
        print("Skip meta-learning: no labeled data.")
    # ===== Save assets =====
    if fisher_matrix is not None:
        model.save_fisher_and_weights(model=meta_encoder, fisher_matrix=fisher_matrix)
        print("save_fisher_and_weights Done.")
    if X_labeled.size > 0:
        model.save_tflite(meta_encoder, tflite_out)
        model.encoder.save("encoder.h5")
        model.classifier.save("classifier.h5")
        model.model.save("meta_model.h5")
        model.model.summary()
        print("save meta tflite Done.")

    # quick forward test
    dummy_x = np.random.rand(1, SEQ_LEN,NUM_FEATURES).astype(np.float32)
    dummy_y = lstm_encoder(dummy_x)
    print("✅ Forward test shape:", dummy_y.shape)
    # -------- Save TFLite / h5 --------
    model.model.save("meta_model.h5")
    MetaModel.save_tflite(model.model, tflite_out)

    print("✅ Pipeline 完成")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder_mode", type=str, default=ENCODER_MODE, choices=["finetune", "freeze", "last_n"])
    parser.add_argument("--load_dir", type=str, default=DATA_DIR)
    parser.add_argument("--meta_out_tf", type=str, default=META_OUT_TF)
    parser.add_argument("--last_n", type=int, default=LAST_N)
    args, _ = parser.parse_known_args()

    ENCODER_MODE = args.encoder_mode
    LOAD_DIR = args.load_dir
    LAST_N = args.last_n
    META_OUT_TF=args.meta_out_tf
    train_pipeline(DATA_DIR, tflite_out=META_OUT_TF)
    #serv_train_tf(META_OUT_TF,num_classes=NUM_CLASSES,seq_len=SEQ_LEN, num_feats=NUM_FEATURES,feature_dim=FEATURE_DIM)
    #serv_pipline(META_OUT_TF, num_classes=NUM_CLASSES,seq_len=SEQ_LEN, num_feats=NUM_FEATURES,feature_dim=FEATURE_DIM)

