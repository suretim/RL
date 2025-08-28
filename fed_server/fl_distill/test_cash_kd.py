import os
import numpy as np
import tensorflow as tf

AUTOTUNE = tf.data.AUTOTUNE

# -------------------------
# KD Loss
# -------------------------
class KDLoss(tf.keras.losses.Loss):
    def __init__(self, temperature=2.0, alpha=0.5):
        super().__init__()
        self.T = float(temperature)
        self.alpha = float(alpha)
        self.ce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.kld = tf.keras.losses.KLDivergence(reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE)

    def call(self, y_true_and_soft, y_pred):
        #y_true, soft = y_true_and_soft
        y_true = y_true_and_soft[0]
        soft   = y_true_and_soft[1]
        ce_loss = self.ce(y_true, y_pred)
        student_log_probs_T = tf.nn.log_softmax(y_pred / self.T, axis=-1)
        kl_loss = self.kld(soft, tf.exp(student_log_probs_T)) * (self.T ** 2)
        return self.alpha * ce_loss + (1 - self.alpha) * kl_loss

# -------------------------
# Load cached soft labels
# -------------------------
def load_softlabel_ds0(npz_path, batch_size=32, shuffle=True):
    arr = np.load(npz_path)
    x, y, soft = arr["x"], arr["y"], arr["soft"]
    ds = tf.data.Dataset.from_tensor_slices(((y, soft), x))
    if shuffle:
        ds = ds.shuffle(len(x), reshuffle_each_iteration=True)
    ds = ds.batch(batch_size).prefetch(AUTOTUNE)
    return ds
def load_softlabel_ds1(npz_path, batch_size, shuffle=True):
    arr = np.load(npz_path)
    x = arr["x"]          # (N, H, W, 3)
    y = arr["y"]          # (N,)
    soft = arr["soft"]    # (N, num_classes)

    ds = tf.data.Dataset.from_tensor_slices((x, y, soft))
    if shuffle:
        ds = ds.shuffle(len(x), reshuffle_each_iteration=True)
    ds = ds.batch(batch_size).prefetch(AUTOTUNE)
    return ds
def load_softlabel_ds(npz_path, batch_size, shuffle=True):
    arr = np.load(npz_path)
    x = arr["x"]
    y = arr["y"]
    soft = arr["soft"]
    # 每个元素: (x, y, soft)
    ds = tf.data.Dataset.from_tensor_slices((x, y, soft))
    if shuffle:
        ds = ds.shuffle(len(x), reshuffle_each_iteration=True)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds

# -------------------------
# Student model
# -------------------------
def build_student(num_classes, img_size=(224,224)):
    inputs = tf.keras.Input(shape=img_size + (3,))
    x = tf.keras.layers.Conv2D(32, 3, strides=2, padding="same", activation="relu")(inputs)
    x = tf.keras.layers.Conv2D(64, 3, strides=2, padding="same", activation="relu")(x)
    x = tf.keras.layers.Conv2D(128, 3, strides=2, padding="same", activation="relu")(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    logits = tf.keras.layers.Dense(num_classes, name="logits")(x)
    return tf.keras.Model(inputs, logits, name="student_cnn")

# -------------------------
# Train
# -------------------------
def train_student(student, train_ds, val_ds, epochs=20, lr=1e-4, temperature=2.0, alpha=0.5):
    loss = KDLoss(temperature=temperature, alpha=alpha)
    student.compile(
        optimizer=tf.keras.optimizers.Adam(lr),
        loss=loss,
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="acc")]
    )
    history = student.fit(train_ds, validation_data=val_ds, epochs=epochs)
    return history

# -------------------------
# Main
# -------------------------
TRAIN_DIR = "../../../../dataset/sprout_y_n_data3/train"  # expected: data/train/class_x/xxx.jpg
VAL_DIR   = "../../../../dataset/sprout_y_n_data3/val"
TEACHER_SAVED_MODEL = "teacher_saved_model"
if __name__ == "__main__":
    cache_train = "./kd_out/soft_train.npz"
    cache_val   = "./kd_out/soft_val.npz"
    batch_size = 32
    img_size = (224,224)
    num_classes = 3  # 根据你的 soft label shape 设置

    # Dataset
    train_ds = load_softlabel_ds(cache_train, batch_size=batch_size, shuffle=True)
    val_ds   = load_softlabel_ds(cache_val, batch_size=batch_size, shuffle=False)

    #train_ds_for_fit = train_ds.map(lambda y_soft_tuple, x: (x, (y_soft_tuple[0], y_soft_tuple[1])))
    #val_ds_for_fit   = val_ds.map(lambda y_soft_tuple, x: (x, (y_soft_tuple[0], y_soft_tuple[1])))
    #train_ds_for_fit = train_ds.map(lambda ys, x: (x, ys))
    #val_ds_for_fit   = val_ds.map(lambda ys, x: (x, ys))
    def map_for_fit(x, y, soft):
        return x, (y, soft)

    train_ds_for_fit = train_ds.map(map_for_fit)
    val_ds_for_fit   = val_ds.map(map_for_fit)

    


    # Student
    student = build_student(num_classes=num_classes, img_size=img_size)
    student.summary()

    # Train
    history = train_student(student, train_ds_for_fit, val_ds_for_fit, epochs=20, lr=1e-4, temperature=2.0, alpha=0.5)

    # Save
    student.save("student_kd.h5")
    print("[OK] Student saved: student_kd.h5")
