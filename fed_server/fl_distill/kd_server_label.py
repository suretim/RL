#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import numpy as np
import tensorflow as tf

AUTOTUNE = tf.data.AUTOTUNE
TRAIN_DIR = "../../../../dataset/sprout_y_n_data3/train"
VAL_DIR   = "../../../../dataset/sprout_y_n_data3/val"
TEACHER_SAVED_MODEL = "teacher_saved_model"

# -------------------------
# Dataset
# -------------------------
def build_datasets(train_dir, val_dir, img_size, batch_size):
    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir, labels="inferred", label_mode="int",
        image_size=img_size, batch_size=batch_size, shuffle=True)
    val_ds = tf.keras.utils.image_dataset_from_directory(
        val_dir, labels="inferred", label_mode="int",
        image_size=img_size, batch_size=batch_size, shuffle=False)
    
    class_names = train_ds.class_names
    num_classes = len(class_names)

    # 0-1 normalization
    def norm(x, y): return (tf.image.convert_image_dtype(x, tf.float32), y)
    train_ds = train_ds.map(norm, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)
    val_ds = val_ds.map(norm, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)
    return train_ds, val_ds, num_classes, class_names

def load_softlabel_ds(npz_path, batch_size, shuffle=True):
    arr = np.load(npz_path)
    x = arr["x"].astype(np.float32) / 255.0
    y = arr["y"].astype(np.int32)
    soft = arr["soft"].astype(np.float32)
    ds = tf.data.Dataset.from_tensor_slices((x, y, soft))
    if shuffle:
        ds = ds.shuffle(len(x), reshuffle_each_iteration=True)
    ds = ds.batch(batch_size).prefetch(AUTOTUNE)
    return ds

# -------------------------
# Student model
# -------------------------
def build_student(num_classes, img_size):
    inputs = tf.keras.Input(shape=img_size + (3,))
    x = tf.keras.layers.Conv2D(32, 3, strides=2, padding="same", activation="relu")(inputs)
    x = tf.keras.layers.Conv2D(64, 3, strides=2, padding="same", activation="relu")(x)
    x = tf.keras.layers.Conv2D(128, 3, strides=2, padding="same", activation="relu")(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    logits = tf.keras.layers.Dense(num_classes, name="logits")(x)
    return tf.keras.Model(inputs, logits, name="student_cnn")

def load_teacher(teacher_path):
    return tf.keras.models.load_model(teacher_path, compile=False)

# -------------------------
# KD Model
# -------------------------
class KDModel(tf.keras.Model):
    def __init__(self, student, temperature=2.0, alpha=0.5):
        super().__init__()
        self.student = student
        self.T = temperature
        self.alpha = alpha
        self.ce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.kld = tf.keras.losses.KLDivergence(reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE)

    def call(self, x, training=False):
        return self.student(x, training=training)

    def train_step(self, data):
        x, y, soft = data
        with tf.GradientTape() as tape:
            logits = self.student(x, training=True)
            ce_loss = self.ce(y, logits)
            student_log_probs_T = tf.nn.log_softmax(logits / self.T, axis=-1)
            kl_loss = self.kld(soft, tf.exp(student_log_probs_T)) * (self.T ** 2)
            loss = self.alpha * ce_loss + (1.0 - self.alpha) * kl_loss
        grads = tape.gradient(loss, self.student.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.student.trainable_variables))
        acc = tf.reduce_mean(
            tf.cast(
                tf.equal(tf.argmax(logits, axis=-1), tf.cast(y, tf.int64)),
                tf.float32
            )
        )


        #acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits, axis=-1), y), tf.float32))
        return {"loss": loss, "acc": acc}

    def test_step(self, data):
        x, y, soft = data
        logits = self.student(x, training=False)
        ce_loss = self.ce(y, logits)
        acc = tf.reduce_mean(
            tf.cast(
                tf.equal(tf.argmax(logits, axis=-1), tf.cast(y, tf.int64)),
                tf.float32
            )
        )
        #acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits, axis=-1), y), tf.float32))
        return {"loss": ce_loss, "acc": acc}

# -------------------------
# Precompute soft labels
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
    print(f"[OK] Saved soft labels: {save_path}, shapes: x{xs.shape}, y{ys.shape}, soft{softs.shape}")

# -------------------------
# Export TFLite
# -------------------------
def export_tflite(model, tflite_fp32_path, tflite_int8_path=None, repdata=None):
    # FP32
    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    tfl = conv.convert()
    with open(tflite_fp32_path, "wb") as f: f.write(tfl)
    print(f"[OK] Saved FP32 TFLite: {tflite_fp32_path}")

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
# python kd_server_label.py --precompute_softlabels --out_dir ./kd_out
# python kd_server_label.py --use_cached_softlabels --epochs 20 --lr 1e-4 --temperature 2.0 --alpha 0.5 --out_dir ./kd_out
# python kd_server_label.py --export_tflite
   
  
# -------------------------
def main(args):
    img_size = (args.img_size[0], args.img_size[1])
    os.makedirs(args.out_dir, exist_ok=True)

    train_ds, val_ds, num_classes, class_names = build_datasets(
        args.train_dir, args.val_dir, img_size, args.batch)

    student = build_student(num_classes=num_classes, img_size=img_size)
    student.summary()

    cache_train = os.path.join(args.out_dir, "soft_train.npz")
    cache_val   = os.path.join(args.out_dir, "soft_val.npz")

    # Precompute soft labels
    if args.precompute_softlabels:
        if not args.teacher_path:
            raise ValueError("--teacher_path required for precompute")
        teacher = load_teacher(args.teacher_path)
        print("[Info] Precomputing soft labels ...")
        precompute_softlabels(train_ds, teacher, args.temperature, cache_train)
        precompute_softlabels(val_ds, teacher, args.temperature, cache_val)
        return

    # Load datasets
    if args.use_cached_softlabels:
        print("[Info] Using cached soft labels ...")
        train_ds = load_softlabel_ds(cache_train, batch_size=args.batch, shuffle=True)
        val_ds   = load_softlabel_ds(cache_val, batch_size=args.batch, shuffle=False)
    else:
        # Online KD: wrap dataset on-the-fly
        if not args.teacher_path:
            raise ValueError("--teacher_path required for online KD")
        teacher = load_teacher(args.teacher_path)
        def make_online(x, y):
            t_logits = teacher(x, training=False)
            soft = tf.nn.softmax(t_logits / args.temperature)
            return x, y, soft
        train_ds = train_ds.map(lambda x, y: tf.py_function(make_online, [x, y], [tf.float32, tf.int32, tf.float32]))
        val_ds = val_ds.map(lambda x, y: tf.py_function(make_online, [x, y], [tf.float32, tf.int32, tf.float32]))

    # KDModel
    kd_model = KDModel(student, temperature=args.temperature, alpha=args.alpha)
    kd_model.compile(optimizer=tf.keras.optimizers.Adam(args.lr))

    # Train
    history = kd_model.fit(train_ds, validation_data=val_ds, epochs=args.epochs)

    # Save student
    stu_h5 = os.path.join(args.out_dir, "student_kd.h5")
    student.save(stu_h5)
    print(f"[OK] Saved student: {stu_h5}")

    # Export TFLite
    if args.export_tflite:
        fp32_path = os.path.join(args.out_dir, "student_kd_fp32.tflite")
        int8_path = os.path.join(args.out_dir, "student_kd_int8.tflite") if args.int8 else None
        rep = rep_dataset_from_dir(args.train_dir, img_size) if args.int8 else None
        export_tflite(student, fp32_path, int8_path, rep)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", type=str, default=TRAIN_DIR)
    parser.add_argument("--val_dir", type=str, default=VAL_DIR)
    parser.add_argument("--img_size", nargs=2, type=int, default=[224,224])
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--teacher_path", type=str, default=TEACHER_SAVED_MODEL)
    parser.add_argument("--precompute_softlabels", action="store_true")
    parser.add_argument("--use_cached_softlabels", action="store_true")
    parser.add_argument("--export_tflite", action="store_true")
    parser.add_argument("--int8", action="store_true")
    parser.add_argument("--out_dir", type=str, default="./kd_out")
    args = parser.parse_args()
    main(args)
