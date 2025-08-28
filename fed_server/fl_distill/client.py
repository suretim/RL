#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Flower Client for Federated KD with Global Soft Labels
- Loads private dataset (train/val) locally.
- Builds a lightweight STUDENT model.
- On each round:
  * Receives server-broadcasted soft labels for the shared PUBLIC dataset.
  * Performs local training:
      loss = alpha * CE(private) + (1 - alpha) * KL(public w/ soft labels)
- Returns updated student weights to server.

Run:
  python client.py --server 127.0.0.1:8080 --train_dir ./data/client1/train --val_dir ./data/client1/val \
                   --public_dir ./public_ds --img_size 224 224 --alpha 0.5 --T 2.0
"""

import argparse
import json
import base64
import numpy as np
import tensorflow as tf
import flwr as fl

AUTOTUNE = tf.data.AUTOTUNE

def b64_to_ndarray(b64: str, shape, dtype):
    arr = np.frombuffer(base64.b64decode(b64.encode("utf-8")), dtype=np.dtype(dtype))
    return arr.reshape(shape)

def build_student(num_classes, img_size):
    inputs = tf.keras.Input(shape=img_size + (3,))
    x = tf.keras.layers.Conv2D(32, 3, strides=2, padding="same", activation="relu")(inputs)
    x = tf.keras.layers.Conv2D(64, 3, strides=2, padding="same", activation="relu")(x)
    x = tf.keras.layers.Conv2D(128, 3, strides=2, padding="same", activation="relu")(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    logits = tf.keras.layers.Dense(num_classes, name="logits")(x)
    return tf.keras.Model(inputs, logits, name="student_cnn")

def load_dir_dataset(data_dir, img_size, batch, shuffle=True):
    ds = tf.keras.utils.image_dataset_from_directory(
        data_dir, labels="inferred", label_mode="int",
        image_size=img_size, batch_size=batch, shuffle=shuffle)
    def norm(x, y): return (tf.image.convert_image_dtype(x, tf.float32), y)
    class_names = ds.class_names
    ds = ds.map(norm, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)
    return ds, class_names

def load_public_dataset(public_dir, img_size, batch):
    # Must be identical to server ordering (shuffle=False)
    ds = tf.keras.utils.image_dataset_from_directory(
        public_dir, labels="inferred", label_mode="int",
        image_size=img_size, batch_size=batch, shuffle=False)
    def norm(x, y): return (tf.image.convert_image_dtype(x, tf.float32), y)
    ds = ds.map(norm, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)
    return ds

class KDLoss(tf.keras.losses.Loss):
    def __init__(self, temperature=2.0, alpha=0.5):
        super().__init__()
        self.T = float(temperature)
        self.alpha = float(alpha)
        self.ce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.kld = tf.keras.losses.KLDivergence(reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE)

    def call(self, y_true_tuple, y_pred_student_logits):
        # y_true_tuple = (y_private, soft_public_probs_or_None, mask_public)
        y_private, soft_public, mask_public = y_true_tuple

        # CE on private
        ce_loss = self.ce(y_private, y_pred_student_logits)

        # KL on public (if present)
        if soft_public is None:
            return ce_loss

        student_log_probs_T = tf.nn.log_softmax(y_pred_student_logits / self.T, axis=-1)
        kl_loss = self.kld(soft_public, tf.exp(student_log_probs_T)) * (self.T ** 2)

        # mask_public==1 rows are public samples (others are private); separate heads
        # 实现上，我们会分两步训练，更清晰：先 private CE，再 public KL。
        # 这里保持接口完整，返回组合损失（若合并训练时使用）。
        return self.alpha * ce_loss + (1.0 - self.alpha) * kl_loss

def train_one_round(student, optimizer, train_private, public_ds, soft_probs, alpha, T, local_epochs):
    # 1) 私有数据：CE
    ce_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    for _ in range(local_epochs):
        for x, y in train_private:
            with tf.GradientTape() as tape:
                logits = student(x, training=True)
                loss = ce_loss(y, logits)
            grads = tape.gradient(loss, student.trainable_variables)
            optimizer.apply_gradients(zip(grads, student.trainable_variables))

    # 2) 公共数据 + 软标签：KL（无私有标签参与）
    if soft_probs is not None:
        T = float(T)
        kld = tf.keras.losses.KLDivergence(reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE)
        idx = 0
        for x, _ in public_ds:
            b = x.shape[0]
            soft_batch = soft_probs[idx:idx+b]
            idx += b
            with tf.GradientTape() as tape:
                logits = student(x, training=True)
                log_p_T = tf.nn.log_softmax(logits / T, axis=-1)
                loss_kd = kld(soft_batch, tf.exp(log_p_T)) * (T ** 2)
            grads = tape.gradient(loss_kd, student.trainable_variables)
            optimizer.apply_gradients(zip(grads, student.trainable_variables))

def evaluate(student, val_private):
    m = tf.keras.metrics.SparseCategoricalAccuracy()
    for x, y in val_private:
        logits = student(x, training=False)
        m.update_state(y, logits)
    return float(m.result().numpy())

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, model, optimizer, train_private, val_private, public_ds, alpha, T, local_epochs):
        self.model = model
        self.optimizer = optimizer
        self.train_private = train_private
        self.val_private = val_private
        self.public_ds = public_ds
        self.alpha = alpha
        self.T = T
        self.local_epochs = local_epochs
        self.soft_cache = None  # will store np.array of public soft labels

    # ---- Flower required methods ----
    def get_parameters(self, config):
        return [w.numpy() for w in self.model.get_weights()]

    def fit(self, parameters, config):
        self.model.set_weights(parameters)

        # Parse KD payload from server
        kd_payload = config.get("kd_payload", None)
        if kd_payload:
            payload = json.loads(kd_payload)
            soft = b64_to_ndarray(payload["soft_b64"], payload["soft_meta"]["shape"], payload["soft_meta"]["dtype"])
            self.soft_cache = soft  # cache for this round
        else:
            self.soft_cache = None

        # Local training
        train_one_round(
            student=self.model,
            optimizer=self.optimizer,
            train_private=self.train_private,
            public_ds=self.public_ds,
            soft_probs=self.soft_cache,
            alpha=self.alpha,
            T=self.T,
            local_epochs=self.local_epochs
        )

        new_params = [w.numpy() for w in self.model.get_weights()]
        num_examples = sum([x.shape[0] for x, _ in self.train_private.unbatch().batch(10)])
        metrics = {"acc_val": evaluate(self.model, self.val_private)}
        return new_params, num_examples, metrics

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        acc = evaluate(self.model, self.val_private)
        num_examples = sum([x.shape[0] for x, _ in self.val_private.unbatch().batch(10)])
        return 0.0, num_examples, {"acc_val": acc}

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--server", type=str, default="127.0.0.1:8080")
    p.add_argument("--train_dir", type=str, required=True)
    p.add_argument("--val_dir", type=str, required=True)
    p.add_argument("--public_dir", type=str, required=True, help="same public dir as server; shuffle must be False")
    p.add_argument("--img_size", nargs=2, type=int, default=[224, 224])
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--epochs_local", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--alpha", type=float, default=0.5, help="CE weight; (1-alpha) for KD")
    p.add_argument("--T", type=float, default=2.0)
    args = p.parse_args()

    img_size = tuple(args.img_size)

    # Datasets
    train_private, class_names = load_dir_dataset(args.train_dir, img_size, args.batch, shuffle=True)
    val_private, _ = load_dir_dataset(args.val_dir, img_size, args.batch, shuffle=False)
    num_classes = len(class_names)

    # Public dataset (must keep same order as server: shuffle=False)
    public_ds = load_public_dataset(args.public_dir, img_size, args.batch)

    # Student & Optimizer
    student = build_student(num_classes=num_classes, img_size=img_size)
    optimizer = tf.keras.optimizers.Adam(args.lr)

    client = FlowerClient(
        model=student,
        optimizer=optimizer,
        train_private=train_private,
        val_private=val_private,
        public_ds=public_ds,
        alpha=args.alpha,
        T=args.T,
        local_epochs=args.epochs_local
    )

    fl.client.start_numpy_client(server_address=args.server, client=client)

if __name__ == "__main__":
    main()
