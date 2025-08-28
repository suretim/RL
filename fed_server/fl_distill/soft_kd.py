#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Soft-Label Knowledge Distillation (TensorFlow/Keras)
- Online KD: teacher(x) -> soft labels on the fly
- Offline KD: precompute and cache teacher soft labels (.npz), then train student without teacher
- Exports student to .h5 and .tflite (optional INT8)
- Supports SavedModel teacher (ViT / CNN)
"""

import os
import argparse
import numpy as np
import tensorflow as tf

AUTOTUNE = tf.data.AUTOTUNE
TRAIN_DIR = "../../../../dataset/sprout_y_n_data3/train"
VAL_DIR   = "../../../../dataset/sprout_y_n_data3/val"
TEACHER_SAVED_MODEL = "teacher_saved_model"
# -------------------------
# 数据加载
# -------------------------

def build_datasets(train_dir, val_dir, img_size=(224, 224), batch_size=32):
    """
    构建训练和验证数据集，同时返回类别数量和 class_names
    """
    # 原始对象，用于获取 class_names
    train_ds_raw = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        labels="inferred",
        label_mode="int",
        image_size=img_size,
        batch_size=batch_size,
        shuffle=True
    )

    val_ds_raw = tf.keras.utils.image_dataset_from_directory(
        val_dir,
        labels="inferred",
        label_mode="int",
        image_size=img_size,
        batch_size=batch_size,
        shuffle=False
    )

    # 获取类别信息
    class_names = train_ds_raw.class_names
    num_classes = len(class_names)

    # 数据归一化
    def norm(x, y):
        x = tf.image.convert_image_dtype(x, tf.float32)
        return x, y

    train_ds = train_ds_raw.map(norm, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)
    val_ds   = val_ds_raw.map(norm, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)

    return train_ds, val_ds, num_classes, class_names


def build_teacher_model(input_shape=(224, 224, 3), num_classes=10):
    base_model = tf.keras.applications.ResNet50(
        include_top=False,
        weights='imagenet',
        input_shape=input_shape,
        pooling='avg'
    )
    base_model.trainable = False  # 冻结特征提取部分
    
    inputs = tf.keras.Input(shape=input_shape)
    x = base_model(inputs, training=False)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    model = tf.keras.Model(inputs, outputs, name="teacher_cnn")
    return model 

# -------------------------
# 学生模型 (轻量 CNN，可替换为自定义 ViT)
# -------------------------
def build_student(num_classes, img_size):
    inputs = tf.keras.Input(shape=img_size + (3,))
    x = inputs
    x = tf.keras.layers.Conv2D(32, 3, strides=2, padding="same", activation="relu")(x)
    x = tf.keras.layers.Conv2D(64, 3, strides=2, padding="same", activation="relu")(x)
    x = tf.keras.layers.Conv2D(128, 3, strides=2, padding="same", activation="relu")(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    logits = tf.keras.layers.Dense(num_classes, name="logits")(x)
    return tf.keras.Model(inputs, logits, name="student_cnn")

# -------------------------
# Teacher 加载
# -------------------------
def load_teacher(teacher_path):
    """
    加载 SavedModel 格式教师模型
    """
    if teacher_path is None or not os.path.exists(teacher_path):
        raise ValueError(f"Teacher path does not exist: {teacher_path}")
    model = tf.keras.models.load_model(teacher_path, compile=False)
    return model

# -------------------------
# KD Loss
# -------------------------
class KDLoss(tf.keras.losses.Loss):
    def __init__(self, temperature=2.0, alpha=0.5, name="kd_loss"):
        super().__init__(name=name)
        self.T = float(temperature)
        self.alpha = float(alpha)
        self.ce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.kld = tf.keras.losses.KLDivergence(reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE)

    def call(self, y_true_and_soft, y_pred_student_logits):
        y_true, soft = y_true_and_soft
        ce_loss = self.ce(y_true, y_pred_student_logits)
        student_log_probs_T = tf.nn.log_softmax(y_pred_student_logits / self.T, axis=-1)
        kl_loss = self.kld(soft, tf.exp(student_log_probs_T)) * (self.T ** 2)
        return self.alpha * ce_loss + (1.0 - self.alpha) * kl_loss

# -------------------------
# Online KD
# -------------------------
def make_online_kd_dataset(dataset, teacher, temperature):
    @tf.function
    def infer_teacher(x):
        t_logits = teacher(x, training=False)
        soft = tf.nn.softmax(t_logits / temperature, axis=-1)
        return soft

    def map_fn(x, y):
        soft = infer_teacher(x)
        return (y, soft), x

    return dataset.map(map_fn, num_parallel_calls=AUTOTUNE)

# -------------------------
# Offline KD
# -------------------------
def precompute_softlabels(dataset, teacher, temperature, save_path):
    xs, ys, softs = [], [], []
    for x, y in dataset:
        t_logits = teacher(x, training=False)
        soft = tf.nn.softmax(t_logits / temperature, axis=-1)
        xs.append(x.numpy()); ys.append(y.numpy()); softs.append(soft.numpy())
    xs = np.concatenate(xs, 0)
    ys = np.concatenate(ys, 0)
    softs = np.concatenate(softs, 0)
    np.savez_compressed(save_path, x=xs, y=ys, soft=softs)
    print(f"[OK] Saved soft labels to: {save_path}  shapes: x{xs.shape}, y{ys.shape}, soft{softs.shape}")

def load_softlabel_ds(npz_path, batch_size, shuffle):
    arr = np.load(npz_path)
    x = arr["x"]; y = arr["y"]; soft = arr["soft"]
    ds = tf.data.Dataset.from_tensor_slices(((y, soft), x))
    if shuffle: ds = ds.shuffle(len(x), reshuffle_each_iteration=True)
    ds = ds.batch(batch_size).prefetch(AUTOTUNE)
    return ds

# -------------------------
# Train & Eval
# -------------------------
def train_student(student, train_ds, val_ds, num_epochs, lr, temperature, alpha, mixed_precision=False):
    if mixed_precision:
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
    loss = KDLoss(temperature=temperature, alpha=alpha)
    student.compile(optimizer=tf.keras.optimizers.Adam(lr), loss=loss,
                    metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="acc")])
    hist = student.fit(train_ds, validation_data=val_ds, epochs=num_epochs)
    return hist

# -------------------------
# TFLite 导出
# -------------------------
def export_tflite(model, tflite_fp32_path, tflite_int8_path=None, repdata=None):
    # FP32
    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    tfl = conv.convert()
    with open(tflite_fp32_path, "wb") as f: f.write(tfl)
    print(f"[OK] Saved FP32 TFLite: {tflite_fp32_path}")

    # INT8
    if tflite_int8_path:
        if repdata is None:
            raise ValueError("INT8 conversion requires representative dataset")
        conv = tf.lite.TFLiteConverter.from_keras_model(model)
        conv.optimizations = [tf.lite.Optimize.DEFAULT]
        conv.representative_dataset = repdata
        conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        conv.inference_input_type = tf.uint8
        conv.inference_output_type = tf.uint8
        tfl = conv.convert()
        with open(tflite_int8_path, "wb") as f: f.write(tfl)
        print(f"[OK] Saved INT8 TFLite: {tflite_int8_path}")

def rep_dataset_from_dir(train_dir, img_size, take=200):
    ds = tf.keras.utils.image_dataset_from_directory(
        train_dir, labels=None, image_size=img_size, batch_size=1, shuffle=True)
    ds = ds.map(lambda x: tf.image.convert_image_dtype(x, tf.float32)).take(take)
    def gen():
        for x in ds:
            yield [tf.cast(x, tf.float32)]
    return gen

# -------------------------
# Main
# -------------------------
def main(args):
    img_size = (args.img_size[0], args.img_size[1])
    os.makedirs(args.out_dir, exist_ok=True)
    train_ds, val_ds, num_classes, class_names = build_datasets(
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        img_size=img_size,
        batch_size=args.batch
    )

    student = build_student(num_classes=num_classes, img_size=img_size)
    student.summary()
    print(f"类名: {class_names}")

    #teacher = build_teacher_model(input_shape=img_size + (3,), num_classes=num_classes)
    #teacher.summary()
    cache_train = os.path.join(args.out_dir, "soft_train.npz")
    cache_val   = os.path.join(args.out_dir, "soft_val.npz")

    if args.precompute_softlabels:
        if not args.teacher_path:
            raise ValueError("--teacher_path is required to precompute soft labels")
        teacher = load_teacher(args.teacher_path)
        print("[Info] Precomputing soft labels ...")
        precompute_softlabels(train_ds, teacher, args.temperature, cache_train)
        precompute_softlabels(val_ds, teacher, args.temperature, cache_val)
        return

    if args.use_cached_softlabels:
        print("[Info] Using cached soft labels.")
        train_kd = load_softlabel_ds(cache_train, args.batch, shuffle=True)
        val_kd   = load_softlabel_ds(cache_val, args.batch, shuffle=False)
    else:
        if not args.teacher_path:
            raise ValueError("--teacher_path is required for online KD")
        teacher = load_teacher(args.teacher_path)
        teacher.summary()
        train_kd = make_online_kd_dataset(train_ds, teacher, args.temperature)
        val_kd   = make_online_kd_dataset(val_ds, teacher, args.temperature)

    history = train_student(
        student=student,
        train_ds=train_kd,
        val_ds=val_kd,
        num_epochs=args.epochs,
        lr=args.lr,
        temperature=args.temperature,
        alpha=args.alpha,
        mixed_precision=args.mixed_precision
    )

    # 保存学生模型
    stu_h5 = os.path.join(args.out_dir, "student_kd.h5")
    student.save(stu_h5)
    print(f"[OK] Saved student: {stu_h5}")

    # 导出 TFLite
    if args.export_tflite:
        fp32_path = os.path.join(args.out_dir, "student_kd_fp32.tflite")
        int8_path = os.path.join(args.out_dir, "student_kd_int8.tflite") if args.int8 else None
        rep = rep_dataset_from_dir(args.train_dir, img_size) if args.int8 else None
        export_tflite(student, fp32_path, int8_path, rep)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--train_dir", type=str, default=TRAIN_DIR)
    p.add_argument("--val_dir", type=str, default=VAL_DIR)
    p.add_argument("--img_size", nargs=2, type=int, default=[224, 224])
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--temperature", type=float, default=2.0)
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--teacher_path", type=str, default=TEACHER_SAVED_MODEL, help="SavedModel folder path")
    p.add_argument("--precompute_softlabels", action="store_true")
    p.add_argument("--use_cached_softlabels", action="store_true")
    p.add_argument("--mixed_precision", action="store_true")
    p.add_argument("--export_tflite", action="store_true")
    p.add_argument("--int8", action="store_true")
    p.add_argument("--out_dir", type=str, default="./kd_out")
    args = p.parse_args()
    main(args)
