import os
import argparse
import numpy as np
import tensorflow as tf

# ==== 默认参数 ====
EPOCHS = 10
BATCH_SIZE = 32
TRAIN_DIR = "../../../../dataset/sprout_y_n_data3/train"
VAL_DIR   = "../../../../dataset/sprout_y_n_data3/val"
SOFT_LABEL_FILE = "soft_labels.npz"
TEMPERATURE = 2.0  # KD 温度
import tensorflow as tf

AUTOTUNE = tf.data.AUTOTUNE  # 定义 AUTOTUNE

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


def build_datasets0(train_dir=TRAIN_DIR, val_dir=VAL_DIR, img_size=(224, 224),batch_size=BATCH_SIZE):
    # 原始对象，用于获取 class_names
    train_ds_raw = tf.keras.utils.image_dataset_from_directory(
        train_dir, labels="inferred", label_mode="int",
        image_size=img_size, batch_size=batch_size, shuffle=True)
    val_ds_raw = tf.keras.utils.image_dataset_from_directory(
        val_dir, labels="inferred", label_mode="int",
        image_size=img_size, batch_size=batch_size, shuffle=False)

    class_names = train_ds_raw.class_names
    num_classes = len(class_names)

    # 简单归一化
    def norm(x, y): return (tf.image.convert_image_dtype(x, tf.float32), y)
    train_ds = train_ds_raw.map(norm, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)
    val_ds = val_ds_raw.map(norm, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)

    return train_ds, val_ds, num_classes, class_names

# -------------------------
# 数据加载
# -------------------------
def load_dataset(data_dir, img_size=(224, 224), batch_size=32):
    ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        image_size=img_size,
        batch_size=batch_size,
        label_mode="int"
    )
    # 归一化到 [0,1]
    ds = ds.map(lambda x, y: (tf.image.convert_image_dtype(x, tf.float32), y))
    return ds

# -------------------------
# 教师模型（EfficientNetB0 或 ViT）
# -------------------------
def create_teacher_model(num_classes, img_size=(224,224)):
    base_model = tf.keras.applications.EfficientNetB0(
        include_top=False, weights="imagenet", input_shape=img_size + (3,), pooling="avg"
    )
    x = tf.keras.layers.Dense(num_classes)(base_model.output)
    # 强制输出为普通 Tensor，避免 EagerTensor
    #outputs = tf.identity(x)
    model = tf.keras.Model(inputs=base_model.input, outputs=x, name="teacher_model")
    return model 

def create_teacher_cnn(input_shape=(224, 224, 3), num_classes=3):
    inputs = tf.keras.Input(shape=input_shape)

    # 卷积提特征
    x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)

    x = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)

    x = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)

    # 全局池化 + 全连接分类
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(512, activation='relu')(x)
    outputs = tf.keras.layers.Dense(num_classes, activation=None)(x)  # logits 输出

    model = tf.keras.Model(inputs, outputs, name="teacher_cnn")
    return model

 
# -------------------------
# 主流程
# (1) python server_teacher_kd.py
# -------------------------
def main(args):
    global EPOCHS, BATCH_SIZE, TRAIN_DIR, VAL_DIR, SOFT_LABEL_FILE
    EPOCHS = args.epochs
    BATCH_SIZE = args.batch
    TRAIN_DIR = args.train_dir
    VAL_DIR = args.val_dir
    SOFT_LABEL_FILE = args.output

    #train_ds = load_dataset(TRAIN_DIR, batch_size=BATCH_SIZE)
    #val_ds = load_dataset(VAL_DIR, batch_size=BATCH_SIZE)
    #train_ds, val_ds, num_classes, class_names = build_datasets(train_dir=TRAIN_DIR,val_dir=VAL_DIR, img_size=(224, 224), batch_size=BATCH_SIZE)
    train_ds, val_ds, num_classes, class_names = build_datasets(
        train_dir=TRAIN_DIR,
        val_dir=VAL_DIR,
        img_size=(224, 224),
        batch_size=BATCH_SIZE
    )



    
    #num_classes = len(train_ds.class_names)
    #teacher = create_teacher_model(num_classes)
    teacher = create_teacher_cnn(input_shape=(224, 224, 3), num_classes=num_classes)

    teacher.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"]
    )

    teacher.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)

    # 保存为 SavedModel 文件夹
    save_dir = "teacher_saved_model"
    os.makedirs(save_dir, exist_ok=True)
    teacher.save(save_dir)
    print(f"[OK] Teacher model saved as SavedModel: {save_dir}")

    # 生成 soft labels
    logits = teacher.predict(train_ds)
    soft_labels = tf.nn.softmax(logits / TEMPERATURE).numpy()
    np.savez(SOFT_LABEL_FILE, soft_labels=soft_labels)
    print(f"Soft labels saved to {SOFT_LABEL_FILE}")

# -------------------------
# 命令行参数
# -------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch", type=int, default=BATCH_SIZE)
    parser.add_argument("--train_dir", type=str, default=TRAIN_DIR)
    parser.add_argument("--val_dir", type=str, default=VAL_DIR)
    parser.add_argument("--output", type=str, default=SOFT_LABEL_FILE)
    args = parser.parse_args()
    main(args)
