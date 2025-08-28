import os
import argparse
import numpy as np
import tensorflow as tf

# ==== 默认参数 ====
EPOCHS = 10
BATCH_SIZE = 32
STUDENT_FEATURE_DIM = 64
#TRAIN_DIR = "./train_data"
#VAL_DIR = "./val_data"
TRAIN_DIR = "../../../../dataset/sprout_y_n_data3/train"  # expected: data/train/class_x/xxx.jpg
VAL_DIR   = "../../../../dataset/sprout_y_n_data3/val"

SOFT_LABEL_FILE = "soft_labels.npz"

def load_dataset(data_dir, img_size=(224, 224), batch_size=32):
    return tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        image_size=img_size,
        batch_size=batch_size,
        label_mode="int"
    )

# 学生模型（轻量 CNN Encoder + 分类头）
def create_student_model(feature_dim, num_classes):
    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = tf.keras.layers.Conv2D(16, 3, activation="relu")(inputs)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    features = tf.keras.layers.Dense(feature_dim, activation="relu")(x)
    logits = tf.keras.layers.Dense(num_classes)(features)  # 输出 logits
    return tf.keras.Model(inputs, logits)

# 蒸馏损失
def distillation_loss(y_true, y_pred, soft_labels, alpha=0.5, temperature=2.0):
    hard_loss = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred, from_logits=True)
    soft_loss = tf.keras.losses.categorical_crossentropy(
        soft_labels, tf.nn.softmax(y_pred / temperature), from_logits=False
    )
    return alpha * hard_loss + (1 - alpha) * soft_loss
# python client_student.py
def main(args):
    # 更新全局变量
    global EPOCHS, BATCH_SIZE, STUDENT_FEATURE_DIM, TRAIN_DIR, VAL_DIR, SOFT_LABEL_FILE
    EPOCHS = args.epochs
    BATCH_SIZE = args.batch
    STUDENT_FEATURE_DIM = args.student_dim
    TRAIN_DIR = args.train_dir
    VAL_DIR = args.val_dir
    SOFT_LABEL_FILE = args.soft_labels

    train_ds = load_dataset(TRAIN_DIR, batch_size=BATCH_SIZE)
    val_ds = load_dataset(VAL_DIR, batch_size=BATCH_SIZE)

    num_classes = len(train_ds.class_names)
    student = create_student_model(STUDENT_FEATURE_DIM, num_classes)

    soft_labels = np.load(SOFT_LABEL_FILE)["soft_labels"]

    optimizer = tf.keras.optimizers.Adam()

    for epoch in range(EPOCHS):
        print(f"Epoch {epoch+1}/{EPOCHS}")
        for step, (x_batch, y_batch) in enumerate(train_ds):
            with tf.GradientTape() as tape:
                logits = student(x_batch, training=True)
                loss = distillation_loss(y_batch, logits, soft_labels[step*BATCH_SIZE:(step+1)*BATCH_SIZE])
            grads = tape.gradient(loss, student.trainable_variables)
            optimizer.apply_gradients(zip(grads, student.trainable_variables))
            if step % 10 == 0:
                print(f"Step {step}, Loss: {tf.reduce_mean(loss).numpy():.4f}")

    student.save("student_model.h5")
    print("Student model saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch", type=int, default=BATCH_SIZE)
    parser.add_argument("--student_dim", type=int, default=STUDENT_FEATURE_DIM)
    parser.add_argument("--train_dir", type=str, default=TRAIN_DIR)
    parser.add_argument("--val_dir", type=str, default=VAL_DIR)
    parser.add_argument("--soft_labels", type=str, default=SOFT_LABEL_FILE)
    args = parser.parse_args()
    main(args)
