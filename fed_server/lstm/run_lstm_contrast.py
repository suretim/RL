import os
import glob
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models

# -----------------------
# 参数
# -----------------------
DATA_GLOB = "./data/*.csv"  # 数据路径
SEQ_LEN = 64
FEATURE_DIM = 64
BATCH_SIZE = 32
EPOCHS_CONTRASTIVE = 10
EPOCHS_CLASSIFIER = 20

# -----------------------
# 1. 加载 CSV 数据
# -----------------------
X_labeled_list, y_labeled_list = [], []
X_unlabeled_list = []

for file in glob.glob(DATA_GLOB):
    df = pd.read_csv(file).fillna(-1)  # NaN 当作无标签
    data = df.values.astype(np.float32)
    
    for i in range(len(data) - SEQ_LEN + 1):
        window = data[i:i+SEQ_LEN, :-1]
        label = data[i+SEQ_LEN-1, -1]
        if label == -1:  # 无标签
            X_unlabeled_list.append(window)
        else:           # 有标签
            X_labeled_list.append(window)
            y_labeled_list.append(int(label))

X_labeled = np.array(X_labeled_list)
y_labeled = np.array(y_labeled_list)
X_unlabeled = np.array(X_unlabeled_list)

print("有标签样本:", X_labeled.shape)
print("无标签样本:", X_unlabeled.shape)

# -----------------------
# 2. 对比学习辅助函数
# -----------------------
def augment_window(x):
    return x + np.random.normal(0, 0.01, x.shape)

def make_contrastive_pairs(X):
    anchors, positives = [], []
    for x in X:
        anchors.append(x)
        positives.append(augment_window(x))
    return np.stack(anchors), np.stack(positives)

class ContrastiveLoss(tf.keras.losses.Loss):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature

    def call(self, z_i, z_j):
        z_i = tf.math.l2_normalize(z_i, axis=1)
        z_j = tf.math.l2_normalize(z_j, axis=1)
        logits = tf.matmul(z_i, z_j, transpose_b=True) / self.temperature
        labels = tf.range(tf.shape(z_i)[0])
        loss_i = tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)
        loss_j = tf.keras.losses.sparse_categorical_crossentropy(labels, tf.transpose(logits), from_logits=True)
        return tf.reduce_mean(loss_i + loss_j)

# -----------------------
# 3. LSTM 编码器
# -----------------------
NUM_FEATS = X_labeled.shape[2] if len(X_labeled) > 0 else 10  # 没有有标签时默认10
lstm_encoder = models.Sequential([
    layers.Input(shape=(SEQ_LEN, NUM_FEATS)),
    layers.LSTM(FEATURE_DIM, return_sequences=False),
    layers.Dense(FEATURE_DIM, activation='relu')
])

# -----------------------
# 4. 对比学习训练（可选）
# -----------------------
if len(X_unlabeled) == 0:
    print("没有无标签数据，生成随机数据用于对比学习")
    X_unlabeled = np.random.randn(100, SEQ_LEN, NUM_FEATS).astype(np.float32)

anchors, positives = make_contrastive_pairs(X_unlabeled)
dataset = tf.data.Dataset.from_tensor_slices((anchors, positives)).shuffle(1024).batch(BATCH_SIZE)
optimizer = tf.keras.optimizers.Adam()

for epoch in range(EPOCHS_CONTRASTIVE):
    for a, p in dataset:
        with tf.GradientTape() as tape:
            z_a = lstm_encoder(a, training=True)
            z_p = lstm_encoder(p, training=True)
            loss = ContrastiveLoss()(z_a, z_p)
        grads = tape.gradient(loss, lstm_encoder.trainable_variables)
        optimizer.apply_gradients(zip(grads, lstm_encoder.trainable_variables))
    print(f"Epoch {epoch+1}/{EPOCHS_CONTRASTIVE}, loss={loss.numpy():.4f}")

# -----------------------
# 5. 有监督特征 + 分类头训练
# -----------------------
if len(X_labeled) > 0:
    features_labeled = lstm_encoder.predict(X_labeled)
    classifier = models.Sequential([
        layers.Input(shape=(FEATURE_DIM,)),
        layers.Dense(32, activation='relu'),
        layers.Dense(3, activation='softmax')
    ])
    classifier.compile(optimizer='adam',
                       loss='sparse_categorical_crossentropy',
                       metrics=['accuracy'])
    classifier.fit(features_labeled, y_labeled,
                   batch_size=BATCH_SIZE,
                   epochs=EPOCHS_CLASSIFIER,
                   validation_split=0.2)

# -----------------------
# 6. TFLite 导出
# -----------------------
def save_tflite(model, out_path):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS,
                                           tf.lite.OpsSet.SELECT_TF_OPS]
    converter._experimental_lower_tensor_list_ops = False
    tflite_model = converter.convert()
    with open(out_path, "wb") as f:
        f.write(tflite_model)
    print("Saved TFLite model:", out_path)

save_tflite(lstm_encoder, "lstm_encoder.tflite")
if len(X_labeled) > 0:
    save_tflite(classifier, "classifier.tflite")
