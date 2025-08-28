import os
import argparse
import numpy as np
import tensorflow as tf
#python teacher_softlabel.py --val_dir data/val --teacher_model teacher.h5

# ===== 默认参数（会被 argparse 覆盖） =====
EPOCHS = 1
BATCH_SIZE = 32
STUDENT_FEATURE_DIM = 64
TRAIN_DIR = "../../../../dataset/sprout_y_n_data3/train"  # expected: data/train/class_x/xxx.jpg
VAL_DIR   = "../../../../dataset/sprout_y_n_data3/val"

#TRAIN_DIR = "data/train"
#VAL_DIR = "data/val"
SOFTLABEL_OUTPUT = "soft_labels.npy"
TEACHER_MODEL_PATH = "teacher.h5"

def load_data(data_dir, img_size=(224, 224), batch_size=32):
    """加载图像数据"""
    ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        image_size=img_size,
        batch_size=batch_size,
        shuffle=False
    )
    ds = ds.map(lambda x, y: (x/255.0, y))
    return ds

def generate_soft_labels(teacher_model, dataset):
    """用教师模型生成 soft labels"""
    soft_labels = []
    hard_labels = []
    for x, y in dataset:
        logits = teacher_model(x, training=False)  # 未经过 softmax 的输出
        probs = tf.nn.softmax(logits, axis=-1)     # 转为 soft label 概率
        soft_labels.append(probs.numpy())
        hard_labels.append(y.numpy())
    return np.vstack(soft_labels), np.concatenate(hard_labels)

def main(args):
    print(f"[INFO] 加载教师模型: {TEACHER_MODEL_PATH}")
    teacher_model = tf.keras.models.load_model(TEACHER_MODEL_PATH)

    print(f"[INFO] 加载验证集: {VAL_DIR}")
    val_ds = load_data(VAL_DIR, batch_size=BATCH_SIZE)

    print("[INFO] 生成 soft labels...")
    soft_labels, hard_labels = generate_soft_labels(teacher_model, val_ds)

    print(f"[INFO] 保存 soft labels 到 {SOFTLABEL_OUTPUT}")
    np.save(SOFTLABEL_OUTPUT, {"soft": soft_labels, "hard": hard_labels})
    print("[INFO] 完成")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch", type=int, default=BATCH_SIZE)
    parser.add_argument("--student_dim", type=int, default=STUDENT_FEATURE_DIM)
    parser.add_argument("--train_dir", type=str, default=TRAIN_DIR)
    parser.add_argument("--val_dir", type=str, default=VAL_DIR)
    parser.add_argument("--softlabel_output", type=str, default=SOFTLABEL_OUTPUT)
    parser.add_argument("--teacher_model", type=str, default=TEACHER_MODEL_PATH)
    args = parser.parse_args()

    # 覆盖全局变量
    EPOCHS = args.epochs
    BATCH_SIZE = args.batch
    STUDENT_FEATURE_DIM = args.student_dim
    TRAIN_DIR = args.train_dir
    VAL_DIR = args.val_dir
    SOFTLABEL_OUTPUT = args.softlabel_output
    TEACHER_MODEL_PATH = args.teacher_model

    main(args)
