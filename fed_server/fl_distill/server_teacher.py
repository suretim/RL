import os
import argparse
import numpy as np
import tensorflow as tf

# ==== 默认参数 ====
EPOCHS = 10
BATCH_SIZE = 32
#TRAIN_DIR = "./train_data"
#VAL_DIR = "./val_data"
TRAIN_DIR = "../../../../dataset/sprout_y_n_data3/train"  # expected: data/train/class_x/xxx.jpg
VAL_DIR   = "../../../../dataset/sprout_y_n_data3/val"
SOFT_LABEL_FILE = "soft_labels.npz"

# 加载数据
def load_dataset(data_dir, img_size=(224, 224), batch_size=32):
    return tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        image_size=img_size,
        batch_size=batch_size,
        label_mode="int"
    )

# 教师模型（可以是 ViT / CNN）
def create_teacher_model(num_classes):
    base_model = tf.keras.applications.EfficientNetB0(
        include_top=False, weights="imagenet", input_shape=(224, 224, 3), pooling="avg"
    )
    x = tf.keras.layers.Dense(num_classes)(base_model.output)  # logits
    model = tf.keras.Model(base_model.input, x)
    return model

def main(args):
    # 更新全局变量
    global EPOCHS, BATCH_SIZE, TRAIN_DIR, VAL_DIR, SOFT_LABEL_FILE
    EPOCHS = args.epochs
    BATCH_SIZE = args.batch
    TRAIN_DIR = args.train_dir
    VAL_DIR = args.val_dir
    SOFT_LABEL_FILE = args.output

    train_ds = load_dataset(TRAIN_DIR, batch_size=BATCH_SIZE)
    val_ds = load_dataset(VAL_DIR, batch_size=BATCH_SIZE)

    num_classes = len(train_ds.class_names)
    teacher = create_teacher_model(num_classes)

    teacher.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"]
    )
    teacher.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)
    teacher.save("teacher_saved_model") 

    # 生成 soft labels
    logits = teacher.predict(train_ds)
    soft_labels = tf.nn.softmax(logits / 2.0).numpy()  # 温度 T=2
    np.savez(SOFT_LABEL_FILE, soft_labels=soft_labels)
    print(f"Soft labels saved to {SOFT_LABEL_FILE}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch", type=int, default=BATCH_SIZE)
    parser.add_argument("--train_dir", type=str, default=TRAIN_DIR)
    parser.add_argument("--val_dir", type=str, default=VAL_DIR)
    parser.add_argument("--output", type=str, default=SOFT_LABEL_FILE)
    args = parser.parse_args()
    main(args)
