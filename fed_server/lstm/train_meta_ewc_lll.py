#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
End-to-end LSTM Meta-learning + EWC + Replay pipeline.
- Loads CSVs
- Contrastive pretrain
- Meta-learning (FOMAML)
- EWC regularization
- Replay buffer
- TFLite export
"""

import os, glob, random
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
import argparse
from tensorflow.keras.optimizers import legacy

from util_model import *
# ---------------- Hyperparameters ----------------
# =============================
# Hyperparameters
# =============================
DATA_GLOB = "../../../data/lll_data/*.csv"

NUM_FEATS = 7


ENCODER_MODE = "freeze"  # one of {"finetune","freeze","last_n"}
LAST_N = 1




CONT_IDX = [0,1,2]
HVAC_IDX = [3,4,5,6]

np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)


def load_csvs(data_dir="../../../data/lll_data"):
    X_labeled_list, y_labeled_list, X_unlabeled_list = [], [], []
    files = sorted(glob.glob(os.path.join(data_dir,"*.csv")))
    for f in files:
        df = pd.read_csv(f).fillna(-1)
        data = df.values.astype(np.float32)
        feats, labels = data[:,:-1], data[:,-1]
        for i in range(len(data)-SEQ_LEN+1):
            w_x = feats[i:i+SEQ_LEN]
            if w_x.shape != (SEQ_LEN, NUM_FEATS):
                continue  # 跳过不完整窗口
            w_y = labels[i+SEQ_LEN-1]
            if w_y==-1:
                X_unlabeled_list.append(w_x)
            else:
                X_labeled_list.append(w_x)
                y_labeled_list.append(int(w_y))
    X_unlabeled = np.array(X_unlabeled_list) if X_unlabeled_list else np.random.randn(200,SEQ_LEN,NUM_FEATS).astype(np.float32)
    X_labeled = np.array(X_labeled_list) if X_labeled_list else np.empty((0,SEQ_LEN,NUM_FEATS),dtype=np.float32)
    y_labeled = np.array(y_labeled_list) if y_labeled_list else np.empty((0,),dtype=np.int32)
    return X_unlabeled, X_labeled, y_labeled





# ---------------- End-to-End Serv Pipeline ----------------
def serv(data_dir="../../../data/lll_data", tflite_out="meta_model_lstm.tflite"):
    print("Loading CSV data...")
    X_unlabeled, X_labeled, y_labeled = load_csvs(data_dir)
    print("Unlabeled samples:", len(X_unlabeled))
    print("Labeled samples:", len(X_labeled), "Labels:", set(y_labeled))
    memory = ReplayBuffer()
    meta_model = MetaModel()
    # optimizer = optimizers.Adam(META_LR)

    optimizer = legacy.Adam(META_LR)

    # ---------------- Contrastive Pretrain ----------------
    print("Start contrastive pretrain...")
    anchors, positives = MetaModel.make_contrastive_pairs(X_unlabeled)
    dataset = tf.data.Dataset.from_tensor_slices((anchors, positives)).batch(BATCH_SIZE)
    c_loss = NTXentLoss()
    for ep in range(EPOCHS_CONTRASTIVE):
        epoch_loss = []
        for a, p in dataset:
            with tf.GradientTape() as tape:
                z_a = meta_model.encoder(a, training=True)
                z_p = meta_model.encoder(p, training=True)
                loss = c_loss(z_a, z_p)
            grads = tape.gradient(loss, meta_model.encoder.trainable_variables)
            optimizer.apply_gradients(zip(grads, meta_model.encoder.trainable_variables))
            epoch_loss.append(loss.numpy())
        print(f"[Contrastive] Epoch {ep + 1}/{EPOCHS_CONTRASTIVE}, loss={np.mean(epoch_loss):.4f}")

    # ---------------- Meta-learning ----------------
    print("Start meta-learning...")


    fisher_matrix = MetaModel.compute_fisher_matrix(meta_model.model, X_labeled, y_labeled)
    if fisher_matrix is not None:
        prev_weights = deepcopy(meta_model.model.trainable_variables)
        for ep in range(EPOCHS_META):
            tasks = sample_tasks(X_labeled, y_labeled)
            #prev_weights = deepcopy(meta_model.model.trainable_variables)
            #fisher_matrix = MetaModel.compute_fisher_matrix(meta_model, X_labeled, y_labeled)

            loss, acc, _ = meta_model.outer_update_with_lll(
                memory, meta_model.model, optimizer, tasks,
                prev_weights=prev_weights,
                fisher_matrix=fisher_matrix,
                lambda_ewc=0.4  # 🔑 这里可以调节强度
            )
            print(f"[Meta] Epoch {ep + 1}/{EPOCHS_META}, query_loss={loss:.4f}, query_acc={acc:.4f}")

        MetaModel.save_fisher_and_weights(model=meta_model.model, fisher_matrix=fisher_matrix)


    # ---------------- Export TFLite ----------------
    MetaModel.save_tflite(meta_model.model, tflite_out)
    # ===== Save assets =====



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder_mode", type=str, default=ENCODER_MODE, choices=["finetune", "freeze", "last_n"])
    parser.add_argument("--data_glob", type=str, default=DATA_GLOB)
    parser.add_argument("--last_n", type=int, default=LAST_N)
    args, _ = parser.parse_known_args()

    ENCODER_MODE = args.encoder_mode
    DATA_GLOB = args.data_glob
    LAST_N = args.last_n

    serv()
