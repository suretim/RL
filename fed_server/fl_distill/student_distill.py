import os
import argparse
import numpy as np
import tensorflow as tf
#python student_distill.py --train_dir data/train --softlabel_input soft_labels.npy

# ===== 默认参数（会被 argparse 覆盖） =====
EPOCHS = 10
BATCH_SIZE = 32
STUDENT_FEATURE_DIM = 64
TRAIN_DIR = "../../../../dataset/sprout_y_n_data3/train"  # expected: data/train/class_x/xxx.jpg
VAL_DIR   = "../../../../dataset/sprout_y_n_data3/val"

SOFTLABEL_INPUT = "soft_labels.npy"
STUDENT_MODEL_PATH = "student_distilled.h5"
TEMPERATURE = 4.0
ALPHA = 0.5  # 蒸馏损失占比

def load_data(data_dir, img_size=(224, 224), batch_size=32):
    ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        image_size=img_size,
        batch_size=batch_size,
        shuffle=True
    )
    ds = ds.map(lambda x, y: (x/255.0, y))
    return ds

def create_student_cnn(feature_dim=64, num_classes=10):
    """一个简单的 CNN 学生模型"""
    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = tf.keras.layers.Conv2D(16, 3, activation='relu')(inputs)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Conv2D(32, 3, activation='relu')(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(feature_dim, activation='relu')(x)
    outputs = tf.keras.layers.Dense(num_classes)(x)  # logits
    return tf.keras.Model(inputs, outputs)

def distillation_loss(y_true, y_pred, soft_labels, temperature, alpha):
    """混合硬标签交叉熵和软标签 KL 散度"""
    # 硬标签损失
    hard_loss = tf.keras.losses.sparse_categorical_crossentropy(y_true, tf.nn.softmax(y_pred))
    # 软标签损失（KL 散度）
    p = tf.nn.softmax(soft_labels / temperature)
    q = tf.nn.softmax(y_pred / temperature)
    soft_loss = tf.keras.losses.KLDivergence()(p, q) * (temperature ** 2)
    return alpha * soft_loss + (1 - alpha) * hard_loss

def main(args):
    print("[INFO] 加载训练集和验证集")
    train_ds = load_data(TRAIN_DIR, batch_size=BATCH_SIZE)
    val_ds = load_data(VAL_DIR, batch_size=BATCH_SIZE)

    print(f"[INFO] 加载 soft labels: {SOFTLABEL_INPUT}")
    soft_data = np.load(SOFTLABEL_INPUT, allow_pickle=True).item()
    soft_labels = soft_data["soft"]
    hard_labels = soft_data["hard"]

    num_classes = soft_labels.shape[1]

    print("[INFO] 构建学生模型")
    student = create_student_cnn(feature_dim=STUDENT_FEATURE_DIM, num_classes=num_classes)
    optimizer = tf.keras.optimizers.Adam()

    @tf.function
    def train_step(images, hard_y, soft_y):
        with tf.GradientTape() as tape:
            logits = student(images, training=True)
            loss = distillation_loss(hard_y, logits, soft_y, TEMPERATURE, ALPHA)
        grads = tape.gradient(loss, student.trainable_variables)
        optimizer.apply_gradients(zip(grads, student.trainable_variables))
        return loss

    print("[INFO] 开始训练")
    for epoch in range(EPOCHS):
        total_loss = 0
        count = 0
        for step, (images, hard_y) in enumerate(train_ds):
            if step >= len(soft_labels) // BATCH_SIZE:
                break
            soft_y = soft_labels[step * BATCH_SIZE : (step + 1) * BATCH_SIZE]
            loss = train_step(images, hard_y, soft_y)
            total_loss += tf.reduce_mean(loss)
            count += 1
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss / count:.4f}")

    print(f"[INFO] 保存学生模型到 {STUDENT_MODEL_PATH}")
    student.save(STUDENT_MODEL_PATH)
    print("[INFO] 完成")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch", type=int, default=BATCH_SIZE)
    parser.add_argument("--student_dim", type=int, default=STUDENT_FEATURE_DIM)
    parser.add_argument("--train_dir", type=str, default=TRAIN_DIR)
    parser.add_argument("--val_dir", type=str, default=VAL_DIR)
    parser.add_argument("--softlabel_input", type=str, default=SOFTLABEL_INPUT)
    parser.add_argument("--student_model", type=str, default=STUDENT_MODEL_PATH)
    parser.add_argument("--temperature", type=float, default=TEMPERATURE)
    parser.add_argument("--alpha", type=float, default=ALPHA)
    args = parser.parse_args()

    # 覆盖全局变量
    EPOCHS = args.epochs
    BATCH_SIZE = args.batch
    STUDENT_FEATURE_DIM = args.student_dim
    TRAIN_DIR = args.train_dir
    VAL_DIR = args.val_dir
    SOFTLABEL_INPUT = args.softlabel_input
    STUDENT_MODEL_PATH = args.student_model
    TEMPERATURE = args.temperature
    ALPHA = args.alpha

    main(args)
