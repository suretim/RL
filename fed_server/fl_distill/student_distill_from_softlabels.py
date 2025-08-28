#!/usr/bin/env python3
# Student: load soft labels .npz and train a student CNN with soft-label distillation (KL) + CE
# Usage example:
# python student_distill_from_softlabels.py --train_dir data/train --val_dir data/val \
#   --softlabel_input soft_labels.npz --epochs 20 --batch 32 --temperature 2.0 --alpha 0.5

import os
import argparse
import numpy as np
import tensorflow as tf

# ===== defaults (overwritten by argparse) =====
EPOCHS = 20
BATCH_SIZE = 32
STUDENT_FEATURE_DIM = 64
TRAIN_DIR = "../../../../dataset/sprout_y_n_data3/train"  # expected: data/train/class_x/xxx.jpg
VAL_DIR   = "../../../../dataset/sprout_y_n_data3/val"

#TRAIN_DIR = "data/train"
#VAL_DIR = "data/val"
SOFTLABEL_INPUT = "soft_labels.npz"
STUDENT_MODEL_PATH = "student_distilled.h5"
TEMPERATURE = 2.0
ALPHA = 0.5  # weight for CE (alpha) vs KD (1-alpha)

def load_dataset(data_dir, img_size=(224,224), batch_size=32, shuffle=True):
    ds = tf.keras.utils.image_dataset_from_directory(
        data_dir, labels="inferred", label_mode="int",
        image_size=img_size, batch_size=batch_size, shuffle=shuffle
    )
    # keep values [0,255]; we will scale to [0,1] for student model (assume float32)
    ds = ds.map(lambda x, y: (tf.image.convert_image_dtype(x, tf.float32), y))
    return ds

def build_student(num_classes, img_size=(224,224), feature_dim=64):
    inp = tf.keras.Input(shape=img_size + (3,))
    x = tf.keras.layers.Conv2D(32, 3, strides=2, padding="same", activation="relu")(inp)
    x = tf.keras.layers.Conv2D(64, 3, strides=2, padding="same", activation="relu")(x)
    x = tf.keras.layers.Conv2D(128, 3, strides=2, padding="same", activation="relu")(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    feats = tf.keras.layers.Dense(feature_dim, activation="relu", name="student_features")(x)
    logits = tf.keras.layers.Dense(num_classes, name="logits")(feats)  # logits
    return tf.keras.Model(inp, logits, name="student_cnn")

def kd_loss_fn(hard_labels, student_logits, soft_probs_T, temperature, alpha):
    # hard CE (from logits)
    ce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    ce_loss = ce(hard_labels, student_logits)
    # KD: KL( teacher_soft_probs || student_probs_T ) * T^2
    # student log-probs at T
    student_log_probs_T = tf.nn.log_softmax(student_logits / temperature, axis=-1)
    # KLDivergence expects (p, q) probabilities; we'll use teacher probs p and exp(student_log_probs_T) as q
    kld = tf.keras.losses.KLDivergence(reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE)
    kd_loss = kld(soft_probs_T, tf.exp(student_log_probs_T)) * (temperature ** 2)
    # combined
    return alpha * ce_loss + (1.0 - alpha) * kd_loss

def main(args):
    global EPOCHS, BATCH_SIZE, STUDENT_FEATURE_DIM, TRAIN_DIR, VAL_DIR, SOFTLABEL_INPUT, STUDENT_MODEL_PATH, TEMPERATURE, ALPHA
    EPOCHS = args.epochs
    BATCH_SIZE = args.batch
    STUDENT_FEATURE_DIM = args.student_dim
    TRAIN_DIR = args.train_dir
    VAL_DIR = args.val_dir
    SOFTLABEL_INPUT = args.softlabel_input
    STUDENT_MODEL_PATH = args.student_model
    TEMPERATURE = args.temperature
    ALPHA = args.alpha

    # datasets
    print("[INFO] Loading datasets")
    train_ds = load_dataset(TRAIN_DIR, batch_size=BATCH_SIZE, img_size=(args.img_size[0], args.img_size[1]), shuffle=True)
    val_ds = load_dataset(VAL_DIR, batch_size=BATCH_SIZE, img_size=(args.img_size[0], args.img_size[1]), shuffle=False)

    # load soft labels
    print(f"[INFO] Loading soft labels from {SOFTLABEL_INPUT}")
    arr = np.load(SOFTLABEL_INPUT)
    # expected keys: 'logits', 'soft', 'hard'
    if "soft" in arr:
        soft_all = arr["soft"]  # shape (N, C) probability with temperature applied in teacher step
    elif "logits" in arr:
        # compute soft with temperature if only logits saved
        logits_all = arr["logits"]
        logits_T = logits_all / float(TEMPERATURE)
        maxl = np.max(logits_T, axis=1, keepdims=True)
        exps = np.exp(logits_T - maxl)
        soft_all = exps / np.sum(exps, axis=1, keepdims=True)
    else:
        raise ValueError("soft labels (.npz) must contain 'soft' or 'logits'")

    # hard labels might be present
    if "hard" in arr:
        hard_all = arr["hard"]
    else:
        hard_all = None

    num_samples = soft_all.shape[0]
    num_classes = soft_all.shape[1]
    print(f"[INFO] soft labels: {num_samples} samples, {num_classes} classes")

    # build student
    student = build_student(num_classes, img_size=(args.img_size[0], args.img_size[1]), feature_dim=STUDENT_FEATURE_DIM)
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr)

    # Training loop: we'll iterate over train_ds but align with soft_all by order.
    # Assumption: train_ds order corresponds to soft labels order (user must ensure same dataset/ordering).
    print("[INFO] Starting distillation training")
    for epoch in range(EPOCHS):
        print(f"Epoch {epoch+1}/{EPOCHS}")
        total_loss = 0.0
        steps = 0
        sample_idx = 0
        for batch_images, batch_labels in train_ds:
            B = batch_images.shape[0]
            # gather matching soft labels slice. If soft_all has fewer samples than train set, wrap or clip.
            if sample_idx + B <= num_samples:
                soft_batch = soft_all[sample_idx: sample_idx + B]
                hard_batch_from_soft = hard_all[sample_idx: sample_idx + B] if hard_all is not None else None
            else:
                # if soft labels shorter, take what's available; pad by repeating last
                remain = max(0, num_samples - sample_idx)
                if remain > 0:
                    part = soft_all[sample_idx: sample_idx + remain]
                    pad = np.tile(soft_all[-1:], (B - remain, 1))
                    soft_batch = np.concatenate([part, pad], axis=0)
                    if hard_all is not None:
                        part_h = hard_all[sample_idx: sample_idx + remain]
                        pad_h = np.tile(hard_all[-1:], (B - remain))
                        hard_batch_from_soft = np.concatenate([part_h, pad_h], axis=0)
                    else:
                        hard_batch_from_soft = None
                else:
                    # no soft labels left: use last row replicated
                    soft_batch = np.tile(soft_all[-1:], (B,1))
                    hard_batch_from_soft = np.tile(hard_all[-1:], (B,)) if hard_all is not None else None

            sample_idx += B

            with tf.GradientTape() as tape:
                logits = student(batch_images, training=True)
                # choose hard labels: prefer dataset labels (batch_labels), else use hard_batch_from_soft
                hard_y = batch_labels if batch_labels is not None else (hard_batch_from_soft if hard_batch_from_soft is not None else tf.zeros((B,), dtype=tf.int32))
                loss = kd_loss_fn(hard_y, logits, soft_batch, TEMPERATURE, ALPHA)
            grads = tape.gradient(loss, student.trainable_variables)
            optimizer.apply_gradients(zip(grads, student.trainable_variables))
            total_loss += float(tf.reduce_mean(loss).numpy())
            steps += 1

        avg_loss = total_loss / steps if steps else 0.0
        print(f"[Epoch {epoch+1}] avg_loss={avg_loss:.4f}")

        # optionally evaluate on val_ds
        acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()
        for x_val, y_val in val_ds:
            preds = student(x_val, training=False)
            acc_metric.update_state(y_val, preds)
        print(f"Validation acc: {acc_metric.result().numpy():.4f}")

    # save student
    print(f"[INFO] Saving student to {STUDENT_MODEL_PATH}")
    student.save(STUDENT_MODEL_PATH)
    print("[INFO] Done.")
#python student_distill_from_softlabels.py --train_dir ./data/train --val_dir ./data/val \
#    --softlabel_input public_softlabels.npz --epochs 30 --batch 32 \
#    --student_dim 64 --img_size 224 224 --temperature 2.0 --alpha 0.5

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch", type=int, default=BATCH_SIZE)
    parser.add_argument("--student_dim", type=int, default=STUDENT_FEATURE_DIM)
    parser.add_argument("--train_dir", type=str, default=TRAIN_DIR)
    parser.add_argument("--val_dir", type=str, default=VAL_DIR)
    parser.add_argument("--softlabel_input", type=str, default=SOFTLABEL_INPUT)
    parser.add_argument("--student_model", type=str, default=STUDENT_MODEL_PATH)
    parser.add_argument("--img_size", nargs=2, type=int, default=[224,224])
    parser.add_argument("--temperature", type=float, default=TEMPERATURE)
    parser.add_argument("--alpha", type=float, default=ALPHA)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()
    main(args)
