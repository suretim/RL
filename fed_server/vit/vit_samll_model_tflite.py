import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# 1. 加载ViT特征提取器（用tfhub官方v2版本）
#vit_url = "https://tfhub.dev/google/vit_base_patch16_224/2"
#vit_layer = hub.KerasLayer(vit_url, trainable=False)
vit_url = "https://tfhub.dev/sayakpaul/vit_b16_fe/1"  # 一个确认可用的ViT特征提取模型

vit_layer = hub.KerasLayer(vit_url, trainable=False)
IMG_SIZE = (224, 224)
num_classes = 5  # 举例
BATCH_SIZE = 32
DATA_BASE_DIR = "../../../../dataset/sprout_y_n_data3"

# 构建只输出特征的模型
inputs = tf.keras.Input(shape=IMG_SIZE + (3,))
features = vit_layer(inputs)  # 输出shape一般是(?,768)
model_fe = tf.keras.Model(inputs, features)

# 2. 训练分类头示范
# 假设你已经有训练数据 X_train, y_train
# ViT专用预处理（重要！）
def vit_preprocess(image):
    """ViT模型需要的特殊预处理"""
    image = tf.cast(image, tf.float32) / 255.0  # 归一化到[0,1]
    image = (image - 0.5) * 2.0  # 转换到[-1,1]范围
    return image
# 数据增强配置
train_datagen = ImageDataGenerator(
    preprocessing_function=vit_preprocess,  # 应用ViT预处理
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
)
# 数据流生成
def create_dataflow(directory, augment=False):
    """创建数据流"""
    generator = train_datagen #if augment else val_test_datagen
    return generator.flow_from_directory(
        directory,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=augment  # 仅训练集需要shuffle
    )
train_data = create_dataflow(os.path.join(DATA_BASE_DIR, "train"), augment=True)

# 这里只示范分类头
inputs_head = tf.keras.Input(shape=features.shape[1:])
x = tf.keras.layers.Dense(256, activation='relu')(inputs_head)
outputs_head = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
model_head = tf.keras.Model(inputs_head, outputs_head)

model_head.compile(optimizer='adam',
                   loss='sparse_categorical_crossentropy',
                   metrics=['accuracy'])
def my_generator():
    for i in range(100):
        yield np.random.rand(224,224,3), 0

dataset = tf.data.Dataset.from_generator(
    my_generator,
    output_signature=(
        tf.TensorSpec(shape=(224,224,3), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.int32)
    )
)
# 假数据示范训练（实际用你的特征和标签训练）
X_train_dummy = np.random.rand(100, features.shape[1])
y_train_dummy = np.random.randint(0, num_classes, 100)
model_head.fit(X_train_dummy, y_train_dummy, epochs=5)
#model_fe.fit(dataset, epochs=5)

# 3. 导出分类头TFLite模型（ESP32端部署）
converter = tf.lite.TFLiteConverter.from_keras_model(model_head)
tflite_model = converter.convert()
with open("classifier_head.tflite", "wb") as f:
    f.write(tflite_model)
