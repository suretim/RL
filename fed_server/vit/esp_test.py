import os
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np

# 1. 加载ViT特征提取器（用tfhub官方v2版本）
#vit_url = "https://tfhub.dev/google/vit_base_patch16_224/2"
#vit_layer = hub.KerasLayer(vit_url, trainable=False)
vit_url = "https://tfhub.dev/sayakpaul/vit_b16_fe/1"  # 一个确认可用的ViT特征提取模型

vit_layer = hub.KerasLayer(vit_url, trainable=False)
IMG_SIZE = (224, 224)
num_classes = 3  # 举例

# 构建只输出特征的模型
inputs = tf.keras.Input(shape=IMG_SIZE + (3,))
features = vit_layer(inputs)  # 输出shape一般是(?,768)
model_fe = tf.keras.Model(inputs, features)

# 2. 训练分类头示范
# 假设你已经有训练数据 X_train, y_train

# 这里只示范分类头
inputs_head = tf.keras.Input(shape=features.shape[1:])
x = tf.keras.layers.Dense(256, activation='relu')(inputs_head)
outputs_head = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
model_head = tf.keras.Model(inputs_head, outputs_head)

model_head.compile(optimizer='adam',
                   loss='sparse_categorical_crossentropy',
                   metrics=['accuracy'])

# 假数据示范训练（实际用你的特征和标签训练）
X_train_dummy = np.random.rand(100, features.shape[1])
y_train_dummy = np.random.randint(0, num_classes, 100)
model_head.fit(X_train_dummy, y_train_dummy, epochs=5)

# 3. 导出分类头TFLite模型（ESP32端部署）
converter = tf.lite.TFLiteConverter.from_keras_model(model_head)
tflite_model = converter.convert()
with open("classifier_head.tflite", "wb") as f:
    f.write(tflite_model)

model_path = "classifier_head.tflite"
size = os.path.getsize(model_path)
print(f"模型大小: {size / 1024:.2f} KB")

interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("输入张量信息：", input_details)
print("输出张量信息：", output_details)
inputs = tf.keras.Input(shape=(224,224,3))
features = vit_layer(inputs)
x = tf.keras.layers.Dense(256, activation='relu')(features)
outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
full_model = tf.keras.Model(inputs, outputs)
converter = tf.lite.TFLiteConverter.from_keras_model(full_model)
tflite_model = converter.convert()
with open("classifier_fe.tflite", "wb") as f:
    f.write(tflite_model)
