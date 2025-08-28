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
- Includes Fisher matrix computation and loading for EWC
- V1.0
"""

import os, glob
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
import random
import argparse 
import datetime
import json
from utils import Meta_Model,ReplayBuffer
# =============================
# Hyperparameters
# =============================
DATA_GLOB = "../../../../data/lll_data/*.csv"
SEQ_LEN = 10
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
NUM_FEATS=7
#ENCODER_MODE = finetune  freeze last_n
ENCODER_MODE ="freeze"
LAST_N= 1
FLOWERING_WEIGHT = 2.0  # gradient boost upper bound for flowering-focus

 
# Make_indices no must do, for checking
def make_indices(model_path="meta_lstm_classifier.tflite", header_path="trainable_tensor_indices.h"):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    tensor_details = interpreter.get_tensor_details()

    trainable_indices = []

    for t in tensor_details:
        name = t['name']
        idx  = t['index']
        shape = t['shape']

        # 僅選 hvac_dense  ro meta_dense kernel/bias
        if ("meta_dense" in name or "hvac_dense" in name):
            # 過濾掉 fused/activation tensor
            if "Relu" in name or ";" in name:
                continue

            # bias 一般是 1D，kernel 一般是 2D
            if len(shape) == 1 or len(shape) == 2:
                print(f"Trainable: {name}, index={idx}, shape={shape}")
                trainable_indices.append(idx)

    # 生成 C 头文件
    with open(header_path, "w") as f:
        f.write("#pragma once\n")
        f.write(f"const int trainable_tensor_indices[] = {{{', '.join(map(str, trainable_indices))}}};\n")
        f.write(f"const int trainable_tensor_count = {len(trainable_indices)};\n")
    print("trainable_tensor_indices =", trainable_indices)
    return trainable_indices


# =============================
# 3) LSTM Encoder (unroll=True, continuous only)
# =============================
   
def build_lstm_encoder0(seq_len, num_feats, feature_dim=FEATURE_DIM):
    inp = layers.Input(shape=(seq_len, num_feats), name="encoder_input")
    x_cont = layers.Lambda(lambda z: z[:, :, :3], name="encoder_lambda")(inp)  # [B,T,3]
    x = layers.LSTM(feature_dim, unroll=True, name="encoder_lstm")(x_cont)
    out = layers.Dense(feature_dim, activation="relu", name="encoder_dense")(x)
    return models.Model(inp, out, name="lstm_encoder")
# =============================
# 4) Meta Model (Encoder + HVAC features)
# =============================
def build_meta_model0(encoder, num_classes=NUM_CLASSES):
    inp = layers.Input(shape=(SEQ_LEN, NUM_FEATS), name="meta_input")
    z_enc = encoder(inp)  # [B, FEATURE_DIM]

    # HVAC slice
    hvac = layers.Lambda(lambda z: z[:, :, 3:7], name="hvac_slice")(inp)   # [B,T,4]
    hvac_mean = layers.Lambda(lambda z: tf.reduce_mean(z, axis=1), name="hvac_mean")(hvac)  # [B,4]

    # Toggle rate via abs(diff)
    hvac_shift = layers.Lambda(lambda z: z[:, 1:, :], name="hvac_shift")(hvac)      # [B,T-1,4]
    hvac_prev  = layers.Lambda(lambda z: z[:, :-1, :], name="hvac_prev")(hvac)     # [B,T-1,4]
    hvac_diff  = layers.Lambda(lambda t: tf.abs(t[0] - t[1]), name="hvac_diff")([hvac_shift, hvac_prev])  # [B,T-1,4]
    hvac_toggle_rate = layers.Lambda(lambda z: tf.reduce_mean(z, axis=1), name="hvac_toggle_rate")(hvac_diff)    # [B,4]

    hvac_feat = layers.Concatenate(name="hvac_concat")([hvac_mean, hvac_toggle_rate])  # [B,8]
    hvac_feat = layers.Dense(16, activation="relu", name="hvac_dense")(hvac_feat)

    x = layers.Concatenate(name="encoder_hvac_concat")([z_enc, hvac_feat])
    x = layers.Dense(64, activation="relu", name="meta_dense_64")(x)
    x = layers.Dense(32, activation="relu", name="meta_dense_32")(x)
    out = layers.Dense(num_classes, activation="softmax", name="meta_out")(x)

    return models.Model(inp, out, name="meta_lstm_classifier")


def inner_update0(model, X_support, y_support, lr_inner=INNER_LR):
    with tf.GradientTape() as tape:
        preds_support = model(X_support, training=True)
        loss_support = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(y_support, preds_support))
    grads_inner = tape.gradient(loss_support, model.trainable_variables)
    updated_vars = [w - lr_inner * g for w, g in zip(model.trainable_variables, grads_inner)]
    return updated_vars

# ===== Fisher Matrix Computation for EWC =====
def compute_fisher_matrix0(model, X, y, num_samples=100):
    fisher = [tf.zeros_like(w) for w in model.trainable_variables]
    
    # Sample subset of data
    idx = np.random.choice(len(X), min(num_samples, len(X)), replace=False)
    X_sample = X[idx]
    y_sample = y[idx]
    
    for x, true_label in zip(X_sample, y_sample):
        with tf.GradientTape() as tape:
            prob = model(np.expand_dims(x, axis=0))[0, true_label]
            log_prob = tf.math.log(prob)
        grads = tape.gradient(log_prob, model.trainable_variables)
        fisher = [f + tf.square(g) for f, g in zip(fisher, grads)]
    #print("fisher matrix:",fisher)
    return [f / num_samples for f in fisher]

# ===== Helpers for flowering focus =====
def is_flowering_seq0(x_seq, light_idx=2, th_light=550.0):
    light_mean = float(np.mean(x_seq[:, light_idx]))
    return light_mean >= th_light

def hvac_toggle_score0(x_seq, hvac_slice=slice(3,7), th_toggle=0.15):
    hv = x_seq[:, hvac_slice]  # [T,4]
    if hv.shape[0] < 2:
        return 0.0, False
    diff = np.abs(hv[1:] - hv[:-1])   # [T-1,4]
    rate = float(diff.mean())
    return rate, rate >= th_toggle

def outer_update_with_lll0(memory,meta_model, meta_optimizer, tasks,
                          lr_inner=INNER_LR, replay_weight=REPLAY_WEIGHT,
                          lambda_ewc=LAMBDA_EWC, prev_weights=None, fisher_matrix=None):
    meta_grads = [tf.zeros_like(v) for v in meta_model.trainable_variables]
    query_acc_list, query_loss_list = [], []

    for X_support, y_support, X_query, y_query in tasks:
        orig_vars = [tf.identity(v) for v in meta_model.trainable_variables]

        # inner update
        updated_vars = inner_update(meta_model, X_support, y_support,lr_inner=lr_inner,)
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
            
            # EWC (using Fisher matrix)
            if prev_weights is not None and fisher_matrix is not None:
                ewc_loss = 0.0

                for w, pw, f in zip(meta_model.trainable_variables, prev_weights, fisher_matrix):
                    ewc_loss += tf.reduce_sum(f * tf.square(w - pw))
                loss_total += lambda_ewc * ewc_loss
                #for i, f in enumerate(prev_weights):
                #    print(f"Fisher matrix for variable {i} has shape: {f.shape}")

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

# =============================
# 5) Save/Load Fisher Matrix and Model Weights
# =============================
 

# Example of loading (commented out since we just saved)
# loaded_fisher = load_ewc_assets(meta_model)

# =============================
# 6) TFLite export (BUILTINS only)
# =============================

def set_trainable_layers0(encoder, meta_model, encoder_mode="finetune", last_n=1):
    """
    設定 encoder 與 meta_model 層的 trainable 狀態

    encoder_mode: "finetune" / "freeze" / "last_n"
    last_n: 只有在 encoder_mode="last_n" 才有意義
    """
    # ===== Encoder =====
    if encoder_mode == "finetune":
        for layer in encoder.layers:
            layer.trainable = True
    elif encoder_mode == "freeze":
        for layer in encoder.layers:
            layer.trainable = False
    elif encoder_mode == "last_n":
        for layer in encoder.layers:
            layer.trainable = False
        if last_n is not None:
            for layer in encoder.layers[-last_n:]:
                layer.trainable = True

    # ===== Meta Model =====
    # 假設 meta_model 的 Dense 層都可以單獨 trainable
    for layer in meta_model.layers:
        # 如果是 encoder 的子模型，不改變（由 encoder 管理）
        #if layer.name.startswith("lstm_encoder") or layer.name.startswith("encoder_"):
        #    continue
        # 其他層全部可訓練
        if layer.name.startswith("meta_dense") or layer.name.startswith("hvac_dense"):
            layer.trainable = True
        else:
            layer.trainable = False
    print(f"✅ Encoder mode: {encoder_mode}, last_n={last_n if encoder_mode=='last_n' else 'N/A'}")

    # 列印實際 trainable 層（只列出有權重的）
    print("\n🔎 [Encoder trainable layers]")
    for layer in encoder.layers:
        if layer.trainable_weights:
            print(f"{layer.name:<20} {'✅ trainable' if layer.trainable else '❌ frozen'}")

    print("\n🔎 [Meta model trainable layers]")
    for layer in meta_model.layers:
        if layer.trainable_weights:
            print(f"{layer.name:<20} {'✅ trainable' if layer.trainable else '❌ frozen'}")
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"trainable_layers_{timestamp}.txt"
    
    log_lines = []
    log_lines.append(f"⚙️ Encoder mode: {ENCODER_MODE}, last_n={LAST_N if ENCODER_MODE=='last_n' else 'N/A'}\n")
    
      
    for i, layer in enumerate(meta_model.layers):
        # 只顯示有 trainable 權重的層
        if layer.trainable_weights:
            status = "✅ trainable" if layer.trainable else "❌ frozen"
            line = f"Layer {i:02d}: {layer.name:<20} {status}"
            print(line)
            log_lines.append(line)

    #with open(log_filename, "w", encoding="utf-8") as f:
    #    f.write("\n".join(log_lines))
    
    #print(f"\n📝 trainable 層清單已儲存到 {log_filename}\n")  



def main(args):
    mm=Meta_Model
    mm.serv(encoder_mode=ENCODER_MODE,last_n=LAST_N,data_glob=DATA_GLOB)

if __name__ == "__main__":
    parser = argparse.ArgumentParser() 
    parser.add_argument("--encoder_mode", type=str, default=ENCODER_MODE)
    parser.add_argument("--data_glob", type=str, default=DATA_GLOB)
    parser.add_argument("--last_n", type=int, default=LAST_N)
    #args = parser.parse_args()
    # ============ 解析參數 ============
    args, unknown = parser.parse_known_args()
  
    
    main(args)
