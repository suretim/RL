#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.2
# ---

# %%
# %%
import os
import glob
import numpy as np
import pandas as pd
import tensorflow as tf

# %%
# ========= 可调参数 =========
SEQ_LEN         = 128          # 每个样本的时间步长度
NUM_FEATS       = 3            # 传感器通道数：例如 [temp, humid, light]
LSTM_UNITS      = 64           # LSTM隐藏单元（也是默认的特征维度）
FEATURE_DIM     = 64           # 输出编码特征维度（可与 LSTM_UNITS 相同）
BATCH_SIZE      = 64
EPOCHS          = 20
LR              = 1e-3

# %%
# 是否同时训练一个小分类头（例如健康/异常二分类）
WITH_CLASSIFIER = True
NUM_CLASSES     = 2

# %%
# 数据路径（示例：data/*.csv，每行一条观测）
DATA_GLOB       = "./data/*.csv"
SAVE_DIR        = "./lstm_sensor_out"
os.makedirs(SAVE_DIR, exist_ok=True)

# %%
# ========= 数据加载与切片 =========
def load_csvs(glob_pattern):
    files = sorted(glob.glob(glob_pattern))
    dfs = []
    for f in files:
        df = pd.read_csv(f)
        # 仅取需要的列；若没有 label，则自动跳过
        cols = ["temp","humid","light"]
        X = df[cols].values.astype(np.float32)
        y = None
        if "label" in df.columns:
            y = df["label"].values.astype(np.int32)
        dfs.append((X,y))
    return dfs

# %%
def zscore_norm(x, mean=None, std=None, eps=1e-6):
    if mean is None:
        mean = x.mean(axis=0, keepdims=True)
    if std is None:
        std = x.std(axis=0, keepdims=True)
    std = np.maximum(std, eps)
    return (x - mean) / std, mean, std

# %%
def make_windows(X, y, seq_len=SEQ_LEN, stride=None):
    """将长序列切成滑窗片段；若 y 存在，窗口的 y 取窗口末端的标签（或多数投票，自行改）"""
    if stride is None:
        stride = seq_len // 2
    xs, ys = [], []
    n = len(X)
    for start in range(0, n - seq_len + 1, stride):
        end = start + seq_len
        xs.append(X[start:end])
        if y is not None:
            ys.append(y[end-1])
    xs = np.stack(xs, axis=0).astype(np.float32)
    ys = np.array(ys, dtype=np.int32) if y is not None else None
    return xs, ys

# %%
import glob
import numpy as np
import pandas as pd

def build_dataset_safe(glob_pattern, with_labels=True):
    all_X, all_y = [], []
    files = sorted(glob.glob(glob_pattern))
    if len(files) == 0:
        raise ValueError(f"No files found for pattern: {glob_pattern}")

    for f in files:
        print(f"Processing file: {f}")
        if f.endswith(".csv"):
            # CSV 文件
            df = pd.read_csv(f)
            cols = ["temp", "humid", "light"]
            X = df[cols].values.astype(np.float32)
            y = df["label"].values.astype(np.int32) if (with_labels and "label" in df.columns) else None
        elif f.endswith(".npz"):
            # npz 文件
            data = np.load(f, allow_pickle=True)  # <- 避免 pickled 数据报错
            X = data["X"].astype(np.float32)
            y = data["y"].astype(np.int32) if (with_labels and "y" in data) else None
        else:
            print(f"Skipped unsupported file: {f}")
            continue

        # 缺失值处理（前向填充 + 均值兜底）
        if np.isnan(X).any():
            for c in range(X.shape[1]):
                col = X[:, c]
                idx = np.where(np.isnan(col))[0]
                for i in idx:
                    col[i] = col[i-1] if i > 0 else np.nan
                if np.isnan(col).any():
                    col[np.isnan(col)] = np.nanmean(col)
                X[:, c] = col

        # z-score 归一化
        Xn, _, _ = zscore_norm(X)

        # 滑窗
        xs, ys = make_windows(Xn, y if with_labels else None)
        all_X.append(xs)
        if with_labels and ys is not None:
            all_y.append(ys)

    X_cat = np.concatenate(all_X, axis=0)
    if with_labels and len(all_y) > 0:
        y_cat = np.concatenate(all_y, axis=0)
    else:
        y_cat = None

    print(f"Built dataset: X={X_cat.shape}, y={None if y_cat is None else y_cat.shape}")
    return X_cat, y_cat
 
# %%
# ========= 模型定义（TFLite Micro 友好）=========
def build_lstm_encoder(num_feats=NUM_FEATS, seq_len=SEQ_LEN,
                       lstm_units=LSTM_UNITS, feature_dim=FEATURE_DIM):
    inp = tf.keras.Input(shape=(seq_len, num_feats), name="sensor_seq")
    # 标准 LSTM（tanh/sigmoid），避免使用不被 TFLite Micro 支持的层
    x = tf.keras.layers.LSTM(
        units=lstm_units,
        activation="tanh",
        recurrent_activation="sigmoid",
        return_sequences=False,   # 仅取最后一步
        use_bias=True,
        name="lstm"
    )(inp)
    # 压缩到固定维度的特征向量
    feat = tf.keras.layers.Dense(feature_dim, activation=None, name="feat_dense")(x)
    feat = tf.keras.layers.Lambda(lambda t: tf.identity(t), name="feature")(feat)
    # 编码器模型：输入序列 -> 特征向量
    enc = tf.keras.Model(inp, feat, name="lstm_encoder")
    return enc

# %%
def build_full_model(with_classifier=WITH_CLASSIFIER):
    enc = build_lstm_encoder()
    if not with_classifier:
        return enc
    out = tf.keras.layers.Dense(NUM_CLASSES, activation=None, name="logits")(enc.output)
    return tf.keras.Model(enc.input, [enc.output, out], name="lstm_encoder_head")

# %%
# ========= 训练流程 =========
def main():
    print("Loading data ...")
    X, y = build_dataset_safe(DATA_GLOB, with_labels=WITH_CLASSIFIER)
    print("X:", X.shape, " y:", None if y is None else y.shape)

    # 训练/验证划分
    if WITH_CLASSIFIER and y is not None:
        idx = np.arange(len(X))
        np.random.shuffle(idx)
        split = int(len(X) * 0.8)
        train_idx, val_idx = idx[:split], idx[split:]
        X_train, y_train = X[train_idx], y[train_idx]
        X_val,   y_val   = X[val_idx],   y[val_idx]
    else:
        # 无监督特征提取：全部用于训练（例如对比学习可自行扩展）
        X_train, X_val = X, X[:0]
        y_train, y_val = None, None

    # 构建模型
    model = build_full_model(with_classifier=WITH_CLASSIFIER)
    model.summary()

    # 编译
    losses = {}
    metrics = {}
    if WITH_CLASSIFIER:
        # 任务：同时最小化分类损失（特征分支无监督，不加损失）
        losses["feature"] = None  # 不对特征直接施加损失
        losses["logits"]  = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        metrics["logits"] = [tf.keras.metrics.SparseCategoricalAccuracy()]
        model.compile(
            optimizer=tf.keras.optimizers.Adam(LR),
            loss=losses,
            metrics=metrics
        )
    else:
        # 仅导出编码器，无监督：用自编码/对比损失可自行扩展；这里占位训练（不会真正更新）
        model.compile(optimizer=tf.keras.optimizers.Adam(LR), loss=None)

    # 训练
    if WITH_CLASSIFIER and y_train is not None and len(X_train) > 0:
        history = model.fit(
            X_train, {"logits": y_train},
            validation_data=(X_val, {"logits": y_val}) if len(X_val) > 0 else None,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            shuffle=True
        )
        model.save(os.path.join(SAVE_DIR, "lstm_encoder_head.h5"))
        # 另存“仅编码器”
        enc = tf.keras.Model(model.input, model.get_layer("feature").output, name="lstm_encoder")
        enc.save(os.path.join(SAVE_DIR, "lstm_encoder.h5"))
    else:
        # 仅保存编码器
        enc = model if not WITH_CLASSIFIER else tf.keras.Model(model.input, model.get_layer("feature").output)
        enc.save(os.path.join(SAVE_DIR, "lstm_encoder.h5"))

    # ========= 导出 TFLite（Float32 & Int8）=========
    def save_tflite(keras_model, out_path, quant_int8=False, rep_data=None):
        """
        保存 TFLite 模型（FP32 或 INT8），支持 LSTM 使用 SELECT_TF_OPS。
        """
        converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    
         
        # FP32 TFLite，解决 LSTM TensorList 问题
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,
            tf.lite.OpsSet.SELECT_TF_OPS
        ]
        # 禁用 experimental_lower_tensor_list_ops
        converter._experimental_lower_tensor_list_ops = False
    
        tflite_model = converter.convert()
        with open(out_path, "wb") as f:
            f.write(tflite_model)
        print("Saved:", out_path, " size:", os.path.getsize(out_path)/1024, "KB")

    encoder_keras = tf.keras.models.load_model(os.path.join(SAVE_DIR, "lstm_encoder.h5"))
    
    # FP32
    save_tflite(encoder_keras, os.path.join(SAVE_DIR, "lstm_encoder_fp32.tflite"))
     
 

    # 同时导出带分类头（可选）
    if WITH_CLASSIFIER:
        head_keras = tf.keras.models.load_model(os.path.join(SAVE_DIR, "lstm_encoder_head.h5"))
        save_tflite(head_keras, os.path.join(SAVE_DIR, "lstm_encoder_head_fp32.tflite"))

# %%
if __name__ == "__main__":
    main()
