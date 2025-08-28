import numpy as np
from PIL import Image
import tensorflow as tf

# 加载 encoder 模型（HDF5 格式或重新训练）
encoder = tf.keras.models.load_model("encoder_model.h5")

# 设置图像大小
img_size = (64, 64)

# 定义 compute_prototypes 函数
def compute_prototypes(encoder, dataset):
    embeddings = []
    labels = []
    for images, lbls in dataset:
        emb = encoder(images)
        embeddings.append(emb.numpy())
        labels.append(lbls.numpy())
    embeddings = np.concatenate(embeddings)
    labels = np.concatenate(labels)
    prototypes = {}
    for label in np.unique(labels):
        proto_means = embeddings[labels == label].mean(axis=0)
        prototypes[label] = proto_means
    return prototypes

# 定义原型推理函数
def predict_prototype(encoder, image, prototypes):
    emb = encoder(tf.expand_dims(image, 0)).numpy()[0]
    distances = {k: np.linalg.norm(emb - v) for k, v in prototypes.items()}
    pred = min(distances, key=distances.get)
    return pred, distances

# 加载你的测试图像
img = Image.open("new_leaf.jpg").resize(img_size)
img = np.array(img) / 255.0

# 加载你之前的训练数据集
dataset = tf.keras.preprocessing.image_dataset_from_directory(
    "data",
    labels='inferred',
    label_mode='int',
    image_size=img_size,
    batch_size=32
)
class_names = dataset.class_names

# 计算类原型并进行推理
prototypes = compute_prototypes(encoder, dataset)
pred, dist = predict_prototype(encoder, img, prototypes)

# 输出推理结果
print(f"✅ Predicted class: {class_names[pred]}")
print(f"📏 Distances: {dist}")
