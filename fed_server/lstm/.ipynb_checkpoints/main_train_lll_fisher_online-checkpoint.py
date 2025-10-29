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
# =============================
# Hyperparameters
# =============================
DATA_GLOB = "../../../lll_data/*.csv"
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

np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)
# Index conventions
CONT_IDX = [0, 1, 2]   # temp, humid, light
HVAC_IDX = [3, 4, 5, 6]  # ac, heater, dehum, hum
 

def make_indices(model_path="meta_lstm_classifier.tflite", header_path="trainable_tensor_indices.h"):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    tensor_details = interpreter.get_tensor_details()

    trainable_indices = []

    for t in tensor_details:
        name = t['name']
        idx  = t['index']
        shape = t['shape']

        # 僅選 Dense 層或 hvac_dense 的 kernel/bias
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
  
def generate_trainable_tensor_indices0(model, tflite_model_path, header_path="trainable_tensor_indices.h"):
    variable_names = [v.name for v in model.trainable_variables]
    print("Python trainable variables:")
    for i, name in enumerate(variable_names):
        print(i, name)

    interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
    interpreter.allocate_tensors()
    tensor_details = interpreter.get_tensor_details()

    trainable_tensor_indices = []

    for v_name in variable_names:
        v_name_clean = v_name.split(':')[0]  # 去掉 ":0"
        matched = False

        v_last = v_name_clean.split('/')[-1]  # kernel 或 bias
        for t in tensor_details:
            t_last = t['name'].split('/')[-1]
            if v_last == t_last:
                trainable_tensor_indices.append(t['index'])
                matched = True
                break

        if not matched:
            print(f"Warning: variable {v_name_clean} not found in tflite tensors!")

    print("trainable_tensor_indices =", trainable_tensor_indices)

    # 可选：生成 C 头文件
    with open(header_path, "w") as f:
        f.write("#pragma once\n")
        f.write(f"const int trainable_tensor_indices[] = {{{', '.join(map(str, trainable_tensor_indices))}}};\n")
        f.write(f"const int trainable_tensor_count = {len(trainable_tensor_indices)};\n")
 


def build_csv_data(data_glob):    
    
    # =============================
    # 1) Load CSV -> Sliding windows
    # =============================
    X_labeled_list, y_labeled_list = [], []
    X_unlabeled_list = []
    
    for file in sorted(glob.glob(data_glob)):
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
        X_labeled = np.empty((0, SEQ_LEN, X_unlabeled.shape[2] if X_unlabeled.size > 0 else 7), dtype=np.float32)
        y_labeled = np.empty((0,), dtype=np.int32)
    
    NUM_FEATS = X_labeled.shape[2] if X_labeled.size > 0 else (X_unlabeled.shape[2] if X_unlabeled.size > 0 else 7)
    
    if NUM_FEATS < 7:
        raise ValueError("Expected at least 7 features per timestep: [temp, humid, light, ac, heater, dehum, hum]. Found: %d" % NUM_FEATS)

    # Provide unlabeled fallback if none
    if X_unlabeled.size == 0:
        X_unlabeled = np.random.randn(200, SEQ_LEN, NUM_FEATS).astype(np.float32)
 
    return X_unlabeled,X_labeled, y_labeled
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
# 3) LSTM Encoder (unroll=True, continuous only)
# =============================
   
def build_lstm_encoder(seq_len, num_feats, feature_dim=FEATURE_DIM):
    inp = layers.Input(shape=(seq_len, num_feats), name="encoder_input")
    x_cont = layers.Lambda(lambda z: z[:, :, :3], name="encoder_lambda")(inp)  # [B,T,3]
    x = layers.LSTM(feature_dim, unroll=True, name="encoder_lstm")(x_cont)
    out = layers.Dense(feature_dim, activation="relu", name="encoder_dense")(x)
    return models.Model(inp, out, name="lstm_encoder")
# =============================
# 4) Meta Model (Encoder + HVAC features)
# =============================
def build_meta_model(encoder, num_classes=NUM_CLASSES):
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


# ===== Fisher Matrix Computation for EWC =====
def compute_fisher_matrix(model, X, y, num_samples=100):
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

def outer_update_with_lll(memory,meta_model, meta_optimizer, tasks,
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
 

def save_fisher_and_weights(model, fisher_matrix, save_dir="ewc_assets"):
    """
    model: 已經 trainable 的 meta_model
    fisher_matrix: 對應 model.trainable_variables 的 Fisher matrix (list of tf.Tensor)
    """
    trainable_vars = model.trainable_variables
    weights = [v.numpy() for v in trainable_vars]
    fisher = [f.numpy() for f in fisher_matrix]
    # 假設 weights 已經取得
    #weights = [w.numpy() for w in model.trainable_weights]
    
    # 生成 layer_shapes
    layer_shapes = [list(w.shape) for w in weights]
    
    # 將 layer_shapes 轉成 JSON bytes
    layer_shapes_json = json.dumps(layer_shapes)
    # 寫入檔案 [[8, 16], [16], [80, 64], [64], [64, 32], [32]]
    with open(os.path.join(save_dir,"layer_shapes.json") , "w") as f:
        f.write(layer_shapes_json)
    
    print("layer_shapes.json saved!")
    layer_shapes_bytes = layer_shapes_json.encode('utf-8')
    # 儲存成 .npz
    np.savez(os.path.join(save_dir,"ewc_assets.npz"), *weights, *fisher)
    #ewc_buffer = np.concatenate([trainable_weights_flattened..., fisher_matrices_flattened...])

    print(f"✅ Trainable weights + Fisher matrix saved to {save_dir}")
    print(f"  Total arrays saved: {len(weights) + len(fisher)}")

# 範例
# fisher_matrix = compute_fisher_matrix(meta_model, X_labeled, y_labeled)
# save_fisher_and_weights(meta_model, fisher_matrix, "ewc_assets.npz")


def save_ewc_assets(model, fisher_matrix, save_dir="ewc_assets"):
    os.makedirs(save_dir, exist_ok=True)
    
    # Save model weights
    model.save_weights(os.path.join(save_dir, "model_weights.h5"))
    
    # Save Fisher matrix
    fisher_numpy = [f.numpy() for f in fisher_matrix]
    np.savez(os.path.join(save_dir, "fisher_matrix.npz"), *fisher_numpy)
    
    print(f"EWC assets saved to {save_dir}")

def load_ewc_assets(model, save_dir="ewc_assets"):
    # Load model weights
    model.load_weights(os.path.join(save_dir, "model_weights.h5"))
    
    # Load Fisher matrix
    fisher_data = np.load(os.path.join(save_dir, "fisher_matrix.npz"))
    fisher_matrix = [tf.constant(arr) for arr in fisher_data.values()]
    
    print(f"EWC assets loaded from {save_dir}")
    return model,fisher_matrix

# Example of loading (commented out since we just saved)
# loaded_fisher = load_ewc_assets(meta_model)

# =============================
# 6) TFLite export (BUILTINS only)
# =============================
def save_tflite(model, out_path):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    tflite_model = converter.convert()
    with open(out_path, "wb") as f:
        f.write(tflite_model)
    print("Saved TFLite:", out_path)
    
def set_trainable_layers(encoder, meta_model, encoder_mode="finetune", last_n=1):
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
    # 更新全局变量
    global  ENCODER_MODE , LAST_N 
    ENCODER_MODE = args.encoder_mode 
    LAST_N = args.last_n 
    
    
    try:
        # ======= Train meta-learning ======= 
        # Build models
        X_unlabeled,X_labeled, y_labeled=build_csv_data(data_glob=DATA_GLOB)
        lstm_encoder = build_lstm_encoder(SEQ_LEN, NUM_FEATS, FEATURE_DIM)
        contrastive_opt = tf.keras.optimizers.Adam()
        ntxent = NTXentLoss(temperature=0.2)
        
        
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
        set_trainable_layers(lstm_encoder, meta_model, ENCODER_MODE, LAST_N)

        memory = ReplayBuffer(capacity=REPLAY_CAPACITY)
        meta_loss_history, meta_acc_history = [], []
        prev_weights = None
        fisher_matrix = None
        
        if X_labeled.size > 0:
            # Compute Fisher matrix on initial model
            fisher_matrix = compute_fisher_matrix(meta_model, X_labeled, y_labeled)
            
            for ep in range(EPOCHS_META):
                tasks = sample_tasks(X_labeled, y_labeled)
                loss, acc, prev_weights = outer_update_with_lll(memory=memory,
                    meta_model=meta_model, meta_optimizer=meta_optimizer, tasks=tasks, 
                    lr_inner=INNER_LR,prev_weights=prev_weights, fisher_matrix=fisher_matrix
                )
                meta_loss_history.append(loss)
                meta_acc_history.append(acc)
                print(f"[Meta] Epoch {ep+1}/{EPOCHS_META}, loss={loss:.4f}, acc={acc:.4f}")
        else:
            print("Skip meta-learning: no labeled data.")
        
        # Save assets if we have them
        if fisher_matrix is not None:
            save_fisher_and_weights(model=meta_model, fisher_matrix=fisher_matrix)
            #save_ewc_assets(meta_model, fisher_matrix)


        # Save models
        save_tflite(lstm_encoder, "lstm_encoder_contrastive.tflite")
        if X_labeled.size > 0:
            save_tflite(meta_model, "meta_lstm_classifier.tflite")
            make_indices(model_path="meta_lstm_classifier.tflite")
            #model = tf.keras.models.load_model("your_keras_model.h5")
            #generate_trainable_tensor_indices(meta_model, "meta_lstm_classifier.tflite")

            print("meta_lstm_classifier.tflite Done.")
        
        # ============ 測試跑一次前向 ============
         
        dummy_x = np.random.rand(1, 10, 7).astype(np.float32)
        dummy_y = lstm_encoder(dummy_x)
        print("✅ 測試前向輸出 shape:", dummy_y.shape)


    
    except KeyboardInterrupt:
        print("Skip meta-learning: no labeled data.")



if __name__ == "__main__":
    parser = argparse.ArgumentParser() 
    parser.add_argument("--encoder_mode", type=str, default=ENCODER_MODE) 
    parser.add_argument("--last_n", type=int, default=LAST_N) 
    #args = parser.parse_args()
    # ============ 解析參數 ============
    args, unknown = parser.parse_known_args()
  
    
    main(args)
