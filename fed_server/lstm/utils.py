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
- V1.1 (cleaned & runnable)
"""

import os
import glob
import json
import random
import datetime
from typing import List, Tuple
from copy import deepcopy

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
import keras
DATA_GLOB = "../../../../data/lll_data/*.csv"

NUM_FEATS = 7


ENCODER_MODE = "freeze"  # one of {"finetune","freeze","last_n"}
LAST_N = 1

np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

# Index conventions
CONT_IDX = [0, 1, 2]   # temp, humid, light
HVAC_IDX = [3, 4, 5, 6]  # ac, heater, dehum, hum
REPLAY_CAPACITY = 1000

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
REPLAY_WEIGHT = 0.3
NUM_CLASSES_OLD = 2
NUM_CLASSES_NEW = 3
EPOCHS_OLD = 5
EPOCHS_NEW = 5
LAMBDA_EWC = 1.0
FLOWERING_WEIGHT = 2.0  # gradient boost upper bound for flowering-focus

class ReplayBuffer:
    def __init__(self, capacity: int = REPLAY_CAPACITY):
        self.buffer: List[Tuple[np.ndarray, int]] = []
        self.capacity = capacity
        self.n_seen = 0
    def add(self, X: np.ndarray, y: np.ndarray):
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
    def sample(self, batch_size: int):
        batch_size = min(batch_size, len(self.buffer))
        idxs = np.random.choice(len(self.buffer), batch_size, replace=False)
        X_s, y_s = zip(*[self.buffer[i] for i in idxs])
        return np.array(X_s), np.array(y_s)

# ------------------------------------------------------------
# Contrastive loss (SimCLR-style, pairwise)
# ------------------------------------------------------------

class NTXentLoss(tf.keras.losses.Loss):
    def __init__(self, temperature: float = 0.2):
        super().__init__()
        self.temperature = temperature
    def call(self, z_i: tf.Tensor, z_j: tf.Tensor):
        z_i = tf.math.l2_normalize(z_i, axis=1)
        z_j = tf.math.l2_normalize(z_j, axis=1)
        logits = tf.matmul(z_i, z_j, transpose_b=True) / self.temperature
        labels = tf.range(tf.shape(z_i)[0])
        loss_i = tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)
        loss_j = tf.keras.losses.sparse_categorical_crossentropy(labels, tf.transpose(logits), from_logits=True)
        return tf.reduce_mean(loss_i + loss_j)

# ------------------------------------------------------------
# Meta model
# ------------------------------------------------------------

class MetaModel:
    def __init__(self,num_classes=2, lambda_ewc=0.4, feature_dim: int = FEATURE_DIM):
        self.feature_dim = feature_dim
        self.lambda_ewc = lambda_ewc


        self.old_params = None
        self.fisher = None
        self.encoder = self.build_lstm_encoder(seq_len=SEQ_LEN, num_feats=NUM_FEATS)
        self.model = self.build_meta_model(self.encoder)
        self.num_classes = num_classes
        self.classifier = self.build_classifier(self.encoder, num_classes)
        self.old_params = None
        self.fisher = None
    @staticmethod
    def build_lstm_encoder(seq_len=SEQ_LEN, num_feats=NUM_FEATS, feature_dim=FEATURE_DIM):
        inp = layers.Input(shape=(seq_len, num_feats), dtype=tf.float32)
        # LSTM 展开，unroll=True
        x = layers.LSTM(feature_dim, return_sequences=True, unroll=True)(inp)
        x = layers.LSTM(feature_dim, unroll=True)(x)
        out = layers.Dense(feature_dim, activation='relu')(x)
        return models.Model(inp, out, name="lstm_encoder")

    @staticmethod
    def build_classifier(encoder, num_classes=NUM_CLASSES_NEW):
        inp = layers.Input(shape=(SEQ_LEN, NUM_FEATS))
        z = encoder(inp)
        out = layers.Dense(num_classes, activation='softmax')(z)
        return models.Model(inp, out, name="meta_model")



    @staticmethod
    def ewc_loss(model, old_params, fisher, lambda_ewc=LAMBDA_EWC):
        if old_params is None or fisher is None:
            return 0.0
        loss = 0.0
        for w, w_old, f in zip(model.trainable_variables, old_params, fisher):
            loss += tf.reduce_sum(f * tf.square(w - w_old))
        return lambda_ewc * loss


    @staticmethod
    def save_ewc_assets(model: tf.keras.Model, fisher_matrix: List[tf.Tensor], save_dir: str = "ewc_assets"):
        os.makedirs(save_dir, exist_ok=True)
        model.save_weights(os.path.join(save_dir, "model_weights.h5"))
        fisher_numpy = [f.numpy() for f in fisher_matrix]
        np.savez(os.path.join(save_dir, "fisher_matrix.npz"), *fisher_numpy)
        print(f"EWC assets saved to {save_dir}")

    @staticmethod
    def load_ewc_assets(model: tf.keras.Model, save_dir: str = "ewc_assets"):
        model.load_weights(os.path.join(save_dir, "model_weights.h5"))
        fisher_data = np.load(os.path.join(save_dir, "fisher_matrix.npz"))
        fisher_matrix = [tf.constant(arr) for arr in fisher_data.values()]
        print(f"EWC assets loaded from {save_dir}")
        return model, fisher_matrix



    # ------ Classifier with HVAC features ------
    def build_meta_model(self, encoder: tf.keras.Model, num_classes: int = NUM_CLASSES):
        inp = layers.Input(shape=(SEQ_LEN, NUM_FEATS), name="meta_input")
        z_enc = encoder(inp)  # [B, FEATURE_DIM]

        # HVAC slice
        hvac = layers.Lambda(lambda z: z[:, :, 3:7], name="hvac_slice")(inp)  # [B,T,4]
        hvac_mean = layers.Lambda(lambda z: tf.reduce_mean(z, axis=1), name="hvac_mean")(hvac)  # [B,4]

        # Toggle rate via abs(diff)
        hvac_shift = layers.Lambda(lambda z: z[:, 1:, :], name="hvac_shift")(hvac)  # [B,T-1,4]
        hvac_prev  = layers.Lambda(lambda z: z[:, :-1, :], name="hvac_prev")(hvac)  # [B,T-1,4]
        hvac_diff  = layers.Lambda(lambda t: tf.abs(t[0] - t[1]), name="hvac_diff")([hvac_shift, hvac_prev])  # [B,T-1,4]
        hvac_toggle_rate = layers.Lambda(lambda z: tf.reduce_mean(z, axis=1), name="hvac_toggle_rate")(hvac_diff)    # [B,4]

        hvac_feat = layers.Concatenate(name="hvac_concat")([hvac_mean, hvac_toggle_rate])  # [B,8]
        hvac_feat = layers.Dense(16, activation="relu", name="hvac_dense")(hvac_feat)

        x = layers.Concatenate(name="encoder_hvac_concat")([z_enc, hvac_feat])
        x = layers.Dense(64, activation="relu", name="meta_dense_64")(x)
        x = layers.Dense(32, activation="relu", name="meta_dense_32")(x)
        out = layers.Dense(num_classes, activation="softmax", name="meta_out")(x)

        return models.Model(inp, out, name="meta_lstm_classifier")

    # ------ Inner update (FOMAML) ------
    @staticmethod
    def inner_update(model: tf.keras.Model, X_support: np.ndarray, y_support: np.ndarray, lr_inner: float = INNER_LR):
        with tf.GradientTape() as tape:
            preds_support = model(X_support, training=True)
            loss_support = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(y_support, preds_support))
        grads_inner = tape.gradient(loss_support, model.trainable_variables)
        # Replace None grads if any
        grads_inner = [g if g is not None else tf.zeros_like(v) for g, v in zip(grads_inner, model.trainable_variables)]
        updated_vars = [w - lr_inner * g for w, g in zip(model.trainable_variables, grads_inner)]
        return updated_vars

    # ------ Fisher Matrix for EWC ------
    @staticmethod
    def compute_fisher_matrix0(model: tf.keras.Model, X: np.ndarray, y: np.ndarray, num_samples: int = 100):
        fisher = [tf.zeros_like(w) for w in model.trainable_variables]
        if len(X) == 0:
            return fisher
        idx = np.random.choice(len(X), min(num_samples, len(X)), replace=False)
        X_sample, y_sample = X[idx], y[idx]
        for x, true_label in zip(X_sample, y_sample):
            with tf.GradientTape() as tape:
                prob = model(np.expand_dims(x, axis=0))[0, true_label]
                log_prob = tf.math.log(tf.maximum(prob, 1e-8))
            grads = tape.gradient(log_prob, model.trainable_variables)
            grads = [g if g is not None else tf.zeros_like(v) for g, v in zip(grads, model.trainable_variables)]
            fisher = [f + tf.square(g) for f, g in zip(fisher, grads)]
        return [f / max(1, len(X_sample)) for f in fisher]

    # ------ Flowering period helpers ------
    @staticmethod
    def is_flowering_seq(x_seq: np.ndarray, light_idx: int = 2, th_light: float = 550.0) -> bool:
        light_mean = float(np.mean(x_seq[:, light_idx]))
        return light_mean >= th_light

    @staticmethod
    def hvac_toggle_score(x_seq: np.ndarray, hvac_slice=slice(3, 7), th_toggle: float = 0.15):
        hv = x_seq[:, hvac_slice]  # [T,4]
        if hv.shape[0] < 2:
            return 0.0, False
        diff = np.abs(hv[1:] - hv[:-1])   # [T-1,4]
        rate = float(diff.mean())
        return rate, rate >= th_toggle



    @staticmethod
    def compute_fisher_matrix(model, X, y, num_samples=200, batch_size=16):
        """
        以 EWC 方式計算 Fisher：對真實標籤的 log-prob 做梯度，累加梯度平方的期望。
        回傳一個張量列表，形狀與 model.trainable_variables 對應。
        """
        # 若沒有標註資料，回傳全零
        if X is None or len(X) == 0 or y is None or len(y) == 0:
            return [tf.zeros_like(v) for v in model.trainable_variables]

        # 取樣以加速
        n = len(X)
        k = min(num_samples, n)
        idx = np.random.choice(n, k, replace=False)
        Xs = X[idx]
        ys = y[idx]

        ds = tf.data.Dataset.from_tensor_slices((Xs, ys)).batch(batch_size)
        fisher = [tf.zeros_like(v) for v in model.trainable_variables]
        count = 0

        # 累加每個 batch 的梯度平方
        for xb, yb in ds:
            with tf.GradientTape() as tape:
                logits = model(xb, training=False)  # [B, C]
                log_prob = tf.nn.log_softmax(logits, axis=-1)
                # 取出真實類別的 log_prob，等價於 sum(one_hot * log_prob)
                yb = tf.cast(yb, tf.int32)
                gathered = tf.gather(log_prob, yb, batch_dims=1)  # [B]
                loss = tf.reduce_mean(gathered)

            grads = tape.gradient(loss, model.trainable_variables)
            for i, g in enumerate(grads):
                if g is not None:
                    fisher[i] = fisher[i] + tf.square(g)
            count += 1

        # 取平均
        fisher = [f / float(max(count, 1)) for f in fisher]
        return fisher

    def outer_update_with_lll(
            self,
            memory,
            meta_model,  # 這裡就是 self.model
            meta_optimizer,
            tasks,
            lr_inner=INNER_LR,
            replay_weight=REPLAY_WEIGHT,
            lambda_ewc=LAMBDA_EWC,
            prev_weights=None,  # 上一輪（或上一任務）參考權重
            fisher_matrix=None  # 與 prev_weights 對應的 Fisher
    ):
        """
        FOMAML outer-loop + replay + EWC（使用外部傳入的 prev_weights 與 fisher_matrix）
        """
        # 聚合所有任務的 meta 梯度
        meta_grads = [tf.zeros_like(v) for v in meta_model.trainable_variables]
        query_acc_list, query_loss_list = [], []

        for X_support, y_support, X_query, y_query in tasks:
            # 保存原始權重
            orig_vars = [tf.identity(v) for v in meta_model.trainable_variables]

            # ------- inner update（單步 FOMAML）-------
            with tf.GradientTape() as tape:
                preds_s = meta_model(X_support, training=True)
                loss_s = tf.reduce_mean(
                    tf.keras.losses.sparse_categorical_crossentropy(y_support, preds_s)
                )
            inner_grads = tape.gradient(loss_s, meta_model.trainable_variables)
            inner_grads = [g if g is not None else tf.zeros_like(v)
                           for g, v in zip(inner_grads, meta_model.trainable_variables)]
            updated_vars = [w - lr_inner * g for w, g in zip(meta_model.trainable_variables, inner_grads)]
            for var, upd in zip(meta_model.trainable_variables, updated_vars):
                var.assign(upd)

            # ------- query + replay + EWC -------
            with tf.GradientTape() as tape:
                preds_q = meta_model(X_query, training=True)
                loss_q = tf.reduce_mean(
                    tf.keras.losses.sparse_categorical_crossentropy(y_query, preds_q)
                )
                total_loss = loss_q

                # Replay
                if len(memory) >= 8:
                    X_old, y_old = memory.sample(32)
                    preds_old = meta_model(X_old, training=True)
                    replay_loss = tf.reduce_mean(
                        tf.keras.losses.sparse_categorical_crossentropy(y_old, preds_old)
                    )
                    total_loss = (1.0 - replay_weight) * total_loss + replay_weight * replay_loss

                # EWC
                if (prev_weights is not None) and (fisher_matrix is not None):
                    ewc = 0.0
                    for f, w, w_old in zip(fisher_matrix, meta_model.trainable_variables, prev_weights):
                        # 確保類型一致（某些環境 fisher 可能是 numpy）
                        if not isinstance(f, tf.Tensor):
                            f = tf.convert_to_tensor(f, dtype=w.dtype)
                        ewc += tf.reduce_sum(f * tf.square(w - w_old))
                    total_loss = total_loss + lambda_ewc * ewc

            grads = tape.gradient(total_loss, meta_model.trainable_variables)
            grads = [g if g is not None else tf.zeros_like(v)
                     for g, v in zip(grads, meta_model.trainable_variables)]
            meta_grads = [mg + g / len(tasks) for mg, g in zip(meta_grads, grads)]

            # 記錄查詢表現
            q_acc = tf.reduce_mean(
                tf.cast(tf.equal(tf.argmax(preds_q, axis=1), y_query), tf.float32)
            )
            query_acc_list.append(float(q_acc.numpy()))
            query_loss_list.append(float(loss_q.numpy()))

            # 還原到 inner update 前的權重（FOMAML）
            for var, orig in zip(meta_model.trainable_variables, orig_vars):
                var.assign(orig)

            # 更新記憶重放
            memory.add(X_support, y_support)
            memory.add(X_query, y_query)

        # ------- 執行 outer 更新 -------
        meta_optimizer.apply_gradients(zip(meta_grads, meta_model.trainable_variables))
        # 回傳本輪平均的 query loss/acc，以及一份當前權重（可作為下輪 prev_weights）
        return float(np.mean(query_loss_list)), float(np.mean(query_acc_list)), [tf.identity(v) for v in
                                                                                 meta_model.trainable_variables]


    # ------ Trainable control ------
    @staticmethod
    def set_trainable_layers(encoder: tf.keras.Model, meta_model: tf.keras.Model, encoder_mode: str = "finetune", last_n: int = 1):
        # Encoder
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

        # Classifier
        for layer in meta_model.layers:
            if layer.name.startswith("meta_dense") or layer.name.startswith("hvac_dense"):
                layer.trainable = True
            else:
                # keep encoder subgraph frozen according to encoder_mode
                if layer.name.startswith("lstm_encoder") or layer.name.startswith("encoder_"):
                    pass
                else:
                    layer.trainable = False

        print(f"✅ Encoder mode: {encoder_mode}, last_n={last_n if encoder_mode=='last_n' else 'N/A'}")
        print("\n🔎 [Encoder trainable layers]")
        for layer in encoder.layers:
            if layer.trainable_weights:
                print(f"{layer.name:<20} {'✅ trainable' if layer.trainable else '❌ frozen'}")
        print("\n🔎 [Meta model trainable layers]")
        for layer in meta_model.layers:
            if layer.trainable_weights:
                print(f"{layer.name:<20} {'✅ trainable' if layer.trainable else '❌ frozen'}")

    # ------ Augment + contrastive pairs ------
    @staticmethod
    def augment_window(x: np.ndarray):
        x_aug = x.copy()
        x_aug[:, CONT_IDX] = x[:, CONT_IDX] + np.random.normal(0, 0.01, x[:, CONT_IDX].shape).astype(np.float32)
        return x_aug

    @classmethod
    def make_contrastive_pairs(cls, X: np.ndarray):
        anchors, positives = [], []
        for w in X:
            anchors.append(w)
            positives.append(cls.augment_window(w))
        return np.stack(anchors).astype(np.float32), np.stack(positives).astype(np.float32)

    # ------ Save / Load helpers ------
    @staticmethod
    def save_fisher_and_weights(model: tf.keras.Model, fisher_matrix: List[tf.Tensor], save_dir: str = "ewc_assets"):
        os.makedirs(save_dir, exist_ok=True)
        trainable_vars = model.trainable_variables
        weights = [v.numpy() for v in trainable_vars]
        fisher = [f.numpy() for f in fisher_matrix]
        #layer_shapes = [list(w.shape) for w in weights]
        #with open(os.path.join(save_dir, "layer_shapes.json"), "w") as f:
        #    f.write(json.dumps(layer_shapes))
        np.savez(os.path.join(save_dir, "ewc_assets.npz"), *weights, *fisher)
        print(f"✅ Saved trainable weights + Fisher matrix to {save_dir} (arrays={len(weights) + len(fisher)})")

    @staticmethod
    def save_tflite(model, out_path):
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
        tflite_model = converter.convert()
        with open(out_path, "wb") as f: f.write(tflite_model)
        print("Saved TFLite:", out_path)



    # ------ Service (orchestrates the pipeline) ------
    def serv(self, encoder_mode: str, last_n: int, data_glob: str):
        global ENCODER_MODE, LAST_N, DATA_GLOB, NUM_FEATS
        ENCODER_MODE = encoder_mode
        LAST_N = last_n
        DATA_GLOB = data_glob

        # ===== Load data =====
        X_unlabeled, X_labeled, y_labeled, num_feats = build_csv_data(DATA_GLOB)
        NUM_FEATS = num_feats  # sync global for model input shapes

        # ===== Build encoder =====
        lstm_encoder = self.build_lstm_encoder(seq_len=SEQ_LEN, num_feats=NUM_FEATS, feature_dim=FEATURE_DIM)

        # ===== Contrastive pretraining =====
        contrastive_opt = tf.keras.optimizers.Adam()
        ntxent = NTXentLoss(temperature=0.2)
        anchors, positives = self.make_contrastive_pairs(X_unlabeled)
        contrast_ds = tf.data.Dataset.from_tensor_slices((anchors, positives)).shuffle(2048).batch(BATCH_SIZE)
        for ep in range(EPOCHS_CONTRASTIVE):
            for a, p in contrast_ds:
                with tf.GradientTape() as tape:
                    za = lstm_encoder(a, training=True)
                    zp = lstm_encoder(p, training=True)
                    loss = ntxent(za, zp)
                grads = tape.gradient(loss, lstm_encoder.trainable_variables)
                grads = [g if g is not None else tf.zeros_like(v) for g, v in zip(grads, lstm_encoder.trainable_variables)]
                contrastive_opt.apply_gradients(zip(grads, lstm_encoder.trainable_variables))
            print(f"[Contrastive] Epoch {ep + 1}/{EPOCHS_CONTRASTIVE}, loss={float(loss.numpy()):.4f}")

        # ===== Meta model =====
        meta_model = self.build_meta_model(lstm_encoder, NUM_CLASSES)
        meta_optimizer = tf.keras.optimizers.Adam(META_LR)
        self.set_trainable_layers(lstm_encoder, meta_model, ENCODER_MODE, LAST_N)

        memory = ReplayBuffer(capacity=REPLAY_CAPACITY)
        prev_weights = None
        fisher_matrix = None

        if X_labeled.size > 0:
            fisher_matrix = self.compute_fisher_matrix(meta_model, X_labeled, y_labeled)
            for ep in range(EPOCHS_META):
                tasks = sample_tasks(X_labeled, y_labeled)
                loss, acc, prev_weights = self.outer_update_with_lll(
                    memory=memory,
                    meta_model=meta_model,
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
            self.save_fisher_and_weights(model=meta_model, fisher_matrix=fisher_matrix)
        if X_labeled.size > 0:
            self.save_tflite(meta_model, "meta_lstm_classifier.tflite")
            print("meta_lstm_classifier.tflite Done.")

        # quick forward test
        dummy_x = np.random.rand(1, SEQ_LEN, NUM_FEATS).astype(np.float32)
        dummy_y = lstm_encoder(dummy_x)
        print("✅ Forward test shape:", dummy_y.shape)



# ------------------------------------------------------------
# (Optional) Generate trainable tensor indices for ESP32 usage
# ------------------------------------------------------------

def make_indices(model_path: str = "meta_lstm_classifier.tflite", header_path: str = "trainable_tensor_indices.h"):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    tensor_details = interpreter.get_tensor_details()
    trainable_indices = []
    for t in tensor_details:
        name, idx, shape = t['name'], t['index'], t['shape']
        if ("meta_dense" in name or "hvac_dense" in name):
            if "Relu" in name or ";" in name:
                continue
            if len(shape) in (1, 2):
                print(f"Trainable: {name}, index={idx}, shape={shape}")
                trainable_indices.append(idx)
    with open(header_path, "w") as f:
        f.write("#pragma once\n")
        f.write(f"const int trainable_tensor_indices[] = {{{', '.join(map(str, trainable_indices))}}};\n")
        f.write(f"const int trainable_tensor_count = {len(trainable_indices)};\n")
    print("trainable_tensor_indices =", trainable_indices)
    return trainable_indices




def build_csv_data(data_glob: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Load CSVs and build sliding windows.
    Returns: X_unlabeled, X_labeled, y_labeled, num_feats
    """
    X_labeled_list, y_labeled_list, X_unlabeled_list = [], [], []

    files = sorted(glob.glob(data_glob))
    if not files:
        print(f"⚠️ No CSV files matched: {data_glob}. Using random fallback for unlabeled pretraining.")

    for file in files:
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

    num_feats = X_labeled.shape[2] if X_labeled.size > 0 else (X_unlabeled.shape[2] if X_unlabeled.size > 0 else 7)

    if num_feats < 7:
        raise ValueError(
            "Expected at least 7 features per timestep: [temp, humid, light, ac, heater, dehum, hum]. Found: %d" % num_feats
        )

    if X_unlabeled.size == 0:
        # provide unlabeled fallback if none
        X_unlabeled = np.random.randn(200, SEQ_LEN, num_feats).astype(np.float32)

    return X_unlabeled, X_labeled, y_labeled, num_feats


def sample_tasks(X: np.ndarray, y: np.ndarray, num_tasks: int = NUM_TASKS,
                 support_size: int = SUPPORT_SIZE, query_size: int = QUERY_SIZE):
    tasks = []
    n = len(X)
    if n < support_size + query_size:
        #print(n,support_size, query_size)
        return tasks
        #raise ValueError(f"Not enough labeled samples to build tasks: need {support_size+query_size}, got {n}")
    for _ in range(num_tasks):
        idx = np.random.choice(n, support_size + query_size, replace=False)
        X_support, y_support = X[idx[:support_size]], y[idx[:support_size]]
        X_query, y_query = X[idx[support_size:]], y[idx[support_size:]]
        tasks.append((X_support, y_support, X_query, y_query))
    return tasks

