#!/usr/bin/env python3
"""
Distill ViT-B16 features into a lightweight CNN student.
- Uses TF-Hub vit_b16_fe/1 as teacher (feature extractor).
- Trains student to match teacher features (MSE) + optional classification loss.
- Exports student to .h5 and .tflite (with int8 quantization option).
"""

import os
import argparse
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np

# -------------------------
# Config / Hyperparameters
# -------------------------
#IMG_SIZE = (224, 224)  # ViT expected input
IMG_SIZE = (64, 64)  # ViT expected input
BATCH_SIZE = 32
EPOCHS = 30 
LEARNING_RATE = 1e-4

# Student projection dimension. To perfectly mimic ViT set to 768.
# For MCU deployment you might set this lower (e.g., 64) and train classifier on that.
STUDENT_FEATURE_DIM = 64  #768

# Loss weights
ALPHA_CLS = 1.0   # classification loss weight
BETA_DISTILL = 1.0  # distillation (MSE) loss weight

# Dataset directories (update to your paths)
TRAIN_DIR = "../../../../dataset/sprout_y_n_data3/train"  # expected: data/train/class_x/xxx.jpg
VAL_DIR   = "../../../../dataset/sprout_y_n_data3/val"

# TFLite output paths
TFLITE_FP32 = "student_fp32.tflite"
TFLITE_INT8 = "student_int8.tflite"

# -------------------------
# Utilities: Data loading
# -------------------------
def get_image_datasets(train_dir, val_dir, img_size=IMG_SIZE, batch_size=BATCH_SIZE):
    if os.path.isdir(train_dir):
        train_ds = tf.keras.utils.image_dataset_from_directory(
            train_dir,
            labels='inferred',
            label_mode='int',
            batch_size=batch_size,
            image_size=img_size,
            shuffle=True,
        )
    else:
        train_ds = None

    if os.path.isdir(val_dir):
        val_ds = tf.keras.utils.image_dataset_from_directory(
            val_dir,
            labels='inferred',
            label_mode='int',
            batch_size=batch_size,
            image_size=img_size,
            shuffle=False,
        )
    else:
        val_ds = None

    return train_ds, val_ds

def build_dummy_ds(num_batches=100, batch_size=BATCH_SIZE, img_size=IMG_SIZE, num_classes=3):
    def gen():
        for _ in range(num_batches):
            imgs = np.random.rand(batch_size, img_size[0], img_size[1], 3).astype(np.float32)
            labels = np.random.randint(0, num_classes, size=(batch_size,))
            yield imgs, labels
    ds = tf.data.Dataset.from_generator(
        gen,
        output_signature=(
            tf.TensorSpec(shape=(batch_size, img_size[0], img_size[1], 3), dtype=tf.float32),
            tf.TensorSpec(shape=(batch_size,), dtype=tf.int32),
        )
    )
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds

# Preprocessing: ViT expects images in [0,1] and size 224x224 (TF-Hub model handles no extra mean/std)
def preprocess_for_student(images):
    # images assumed 0-255 input; convert to 0-1 float
     
    images = tf.image.convert_image_dtype(images, dtype=tf.float32)
    return images
def preprocess_for_vit(images):
    # images assumed 0-255 input; convert to 0-1 float
    images = tf.image.resize(images, size=(224, 224))
    images = tf.image.convert_image_dtype(images, dtype=tf.float32)
    return images

# -------------------------
# Teacher (ViT) loader
# -------------------------
def load_vit_teacher(vit_url="https://tfhub.dev/sayakpaul/vit_b16_fe/1"):
    print("Loading ViT feature extractor from TF-Hub:", vit_url)
    vit_layer = hub.KerasLayer(vit_url, trainable=False)
    # Wrap into a Keras Model for convenience
    inputs = tf.keras.Input(shape=IMG_SIZE + (3,))
    #inputs = tf.keras.Input(shape=(224, 224) + (3,))
    
    x = preprocess_for_vit(inputs)
    feats = vit_layer(x)
    feats = tf.keras.layers.Dense(STUDENT_FEATURE_DIM, activation=None)(feats)  # Reducing from 768 to 384

    model = tf.keras.Model(inputs=inputs, outputs=feats, name="vit_teacher_fe")
    return model

# -------------------------
# Student model
# -------------------------
def create_student_cnn(input_shape=IMG_SIZE+(3,), feature_dim=STUDENT_FEATURE_DIM):
    inputs = tf.keras.Input(shape=input_shape)
    x = preprocess_for_student(inputs)  # same preprocessing pipeline: 0-1
    # Lightweight conv stack (tunable)
    x = tf.keras.layers.Conv2D(32, 3, strides=2, padding="same", activation="relu")(x)  # 112x112x32
    x = tf.keras.layers.Conv2D(64, 3, strides=2, padding="same", activation="relu")(x)  # 56x56x64
    x = tf.keras.layers.Conv2D(128, 3, strides=2, padding="same", activation="relu")(x)  # 28x28x128
    x = tf.keras.layers.Conv2D(256, 3, strides=2, padding="same", activation="relu")(x)  # 14x14x256
    x = tf.keras.layers.Conv2D(64, 3, strides=2, padding="same", activation="relu")(x)  # 14x14x256
    #x = tf.keras.layers.GlobalAveragePooling2D()(x)  # (batch, 128)
    x = tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=(1, 1), padding="valid")(x)  # (batch, 7, 7, 128)
    x = tf.keras.layers.Reshape((64,))(x)  # 输出 (batch, 512)
    #x = tf.keras.layers.Flatten()(x)
    # Map to feature_dim to match teacher feature size (or a compressed dim)
    feats = tf.keras.layers.Dense(feature_dim, name="student_features")(x)
    # Optionally L2-normalize or not — depends on distillation objective; leave raw here
    model = tf.keras.Model(inputs=inputs, outputs=feats, name="student_cnn")
    model.summary()
    return model

# -------------------------
# Distiller model (from TF tutorial pattern)
# ------------------------- 

class FeatureDistiller(tf.keras.Model):
    def __init__(self, student, teacher, num_classes=None, alpha_cls=ALPHA_CLS, beta_distill=BETA_DISTILL):
        super().__init__()
        self.student = student
        self.teacher = teacher
        self.alpha_cls = alpha_cls
        self.beta_distill = beta_distill

        # If you want to train classification head jointly, construct one:
        self.num_classes = num_classes
        if num_classes is not None:
            self.classifier = tf.keras.Sequential([
                tf.keras.layers.Input(shape=(student.output_shape[-1],)),
                tf.keras.layers.Dense(64, activation="relu"),
                tf.keras.layers.Dense(num_classes)  # logits
            ], name="student_classifier")
        else:
            self.classifier = None
         
        # Metrics
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.cls_loss_tracker = tf.keras.metrics.Mean(name="cls_loss")
        self.distill_loss_tracker = tf.keras.metrics.Mean(name="distill_loss")
        self.acc = tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy") if num_classes is not None else None

    @property
    def metrics(self):
        metrics = [self.loss_tracker, self.distill_loss_tracker, self.cls_loss_tracker]
        if self.acc is not None:
            metrics.append(self.acc)
        return metrics

    def compile(self, optimizer, cls_loss_fn=None, distill_loss_fn=None):
        super().compile()
        self.optimizer = optimizer
        self.cls_loss_fn = cls_loss_fn or tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.distill_loss_fn = distill_loss_fn or tf.keras.losses.MeanSquaredError()

    def train_step(self, data):
        images, labels = data

        # Teacher forward (no grads)
        teacher_feats = self.teacher(images, training=False)
        
        with tf.GradientTape() as tape:
            student_feats = self.student(images, training=True)
            loss_distill = self.distill_loss_fn(teacher_feats, student_feats)
            if self.classifier is not None:
                logits = self.classifier(student_feats, training=True)
                loss_cls = self.cls_loss_fn(labels, logits)
            else:
                loss_cls = 0.0

            loss = self.alpha_cls * loss_cls + self.beta_distill * loss_distill

        trainable_vars = self.student.trainable_variables + (self.classifier.trainable_variables if self.classifier is not None else [])
        grads = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(grads, trainable_vars))

        # Update metrics
        self.loss_tracker.update_state(loss)
        self.distill_loss_tracker.update_state(loss_distill)
        self.cls_loss_tracker.update_state(loss_cls)
        if self.acc is not None and self.classifier is not None:
            self.acc.update_state(labels, logits)

        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        images, labels = data
        teacher_feats = self.teacher(images, training=False)
        student_feats = self.student(images, training=False)
        loss_distill = self.distill_loss_fn(teacher_feats, student_feats)
        if self.classifier is not None:
            logits = self.classifier(student_feats, training=False)
            loss_cls = self.cls_loss_fn(labels, logits)
            if self.acc is not None:
                self.acc.update_state(labels, logits)
        else:
            loss_cls = 0.0
        loss = self.alpha_cls * loss_cls + self.beta_distill * loss_distill

        self.loss_tracker.update_state(loss)
        self.distill_loss_tracker.update_state(loss_distill)
        self.cls_loss_tracker.update_state(loss_cls)
        return {m.name: m.result() for m in self.metrics}

# -------------------------
# Representative dataset for TFLite quantization
# -------------------------
def representative_data_gen(dataset, num_calib_steps=100):
    # dataset yields (images, labels). We need to yield a single input tensor for converter
    ds = dataset.unbatch().take(num_calib_steps).map(lambda x, y: x)
    for input_value in ds:
        # input_value shape: (H,W,3)
        img = tf.expand_dims(input_value, 0)
        img = tf.cast(img, tf.float32)
        yield [img]

# -------------------------
# Save TFLite functions
# -------------------------
def export_tflite(student_model, example_input_shape=(1, IMG_SIZE[0], IMG_SIZE[1], 3),
                  out_fp=TFLITE_FP32, quantize_int8=False, representative_ds=None):
    # Wrap student to include classifier? For feature-only export we export the student model itself
    print("Converting to TFLite FP32:", out_fp)
    converter = tf.lite.TFLiteConverter.from_keras_model(student_model)
    tflite_model = converter.convert()
    with open(out_fp, "wb") as f:
        f.write(tflite_model)
    print("Saved:", out_fp)

    if quantize_int8:
        if representative_ds is None:
            raise ValueError("Representative dataset required for int8 quantization")
        print("Converting to int8 TFLite:", TFLITE_INT8)
        converter = tf.lite.TFLiteConverter.from_keras_model(student_model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = lambda: representative_data_gen(representative_ds, num_calib_steps=200)
        # For full integer quantization:
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.uint8  # or tf.int8 depending on your deployment
        converter.inference_output_type = tf.uint8
        tflite_int8 = converter.convert()
        with open(TFLITE_INT8, "wb") as f:
            f.write(tflite_int8)
        print("Saved int8:", TFLITE_INT8)

# -------------------------
# Main run
# -------------------------
def main(args):
    # Datasets
    train_ds, val_ds = get_image_datasets(TRAIN_DIR, VAL_DIR)
    if train_ds is None:
        print("Train directory not found; using dummy dataset for quick test.")
        train_ds = build_dummy_ds(num_batches=200)
        val_ds = build_dummy_ds(num_batches=20)
        num_classes = 3
    else:
        # infer num classes from dataset
        num_classes = len(train_ds.class_names)
        # Prefetch/resize pipeline
        train_ds = train_ds.map(lambda x, y: (tf.image.resize(x, IMG_SIZE), y)).prefetch(tf.data.AUTOTUNE)
        val_ds = val_ds.map(lambda x, y: (tf.image.resize(x, IMG_SIZE), y)).prefetch(tf.data.AUTOTUNE)

    # Models
    teacher = load_vit_teacher()
    student = create_student_cnn(feature_dim=STUDENT_FEATURE_DIM)

    # Distiller: include classification head if you want supervised signal
    include_classifier = True  # set False to only do feature-matching distillation
    distiller = FeatureDistiller(student=student, teacher=teacher,
                                 num_classes=num_classes if include_classifier else None,
                                 alpha_cls=ALPHA_CLS, beta_distill=BETA_DISTILL)

    opt = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    distiller.compile(optimizer=opt)

    # Fit
    print("Start distillation training...")
    distiller.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)

    # Save student core model (feature extractor)
    student.save("student_encoder.h5")
    print("Saved student_encoder.h5")

    # Optionally also save student + classifier jointly
    if include_classifier:
        # Build an end-to-end model: inputs -> student -> classifier (logits)
        inp = tf.keras.Input(shape=IMG_SIZE + (3,))
        feats = student(inp)
        logits = distiller.classifier(feats)
        end2end = tf.keras.Model(inputs=inp, outputs=logits, name="student_end2end")
        end2end.save("student_end2end.h5")
        print("Saved student_end2end.h5")

    # Export to TFLite (feature-only student)
    # For FP32
    export_tflite(student, out_fp=TFLITE_FP32, quantize_int8=False)

    # For int8 quantization for MCU (requires representative dataset)
    try:
        export_tflite(student, quantize_int8=True, representative_ds=train_ds)
    except Exception as e:
        print("Int8 quantization failed (representative dataset missing or converter error):", e)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch", type=int, default=BATCH_SIZE)
    parser.add_argument("--student_dim", type=int, default=STUDENT_FEATURE_DIM)
    parser.add_argument("--train_dir", type=str, default=TRAIN_DIR)
    parser.add_argument("--val_dir", type=str, default=VAL_DIR)
    args = parser.parse_args()
    # update globals from args (simple)
    EPOCHS = args.epochs
    BATCH_SIZE = args.batch
    STUDENT_FEATURE_DIM = args.student_dim
    TRAIN_DIR = args.train_dir
    VAL_DIR = args.val_dir

    main(args)
