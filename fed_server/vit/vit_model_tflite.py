import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import os

# 你的 ViT 模型 URL
vit_url = "https://tfhub.dev/sayakpaul/vit_b16_fe/1"

#vit_url = "https://tfhub.dev/google/vit_base_patch16_224/1"
IMG_SIZE = (224, 224)
categories = ["cat", "dog"]  # 你的类别列表

def build_vit_model():
    vit_layer = hub.KerasLayer(
        vit_url,
        trainable=False,
        name='vit_feature_extractor'
    )
    inputs = tf.keras.Input(shape=IMG_SIZE + (3,))
    x = vit_layer(inputs)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    outputs = tf.keras.layers.Dense(len(categories), activation='softmax')(x)
    return tf.keras.Model(inputs, outputs)

# 构建并加载训练好的权重
model = build_vit_model()
# model.load_weights("your_weights.h5")  # 如果有训练好的权重就加载

# 先编译一次（只是为了模型能运行）
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 运行一次推理，让 SavedModel 有输入输出签名
dummy_x = np.random.rand(1, IMG_SIZE[0], IMG_SIZE[1], 3).astype(np.float32)
model.predict(dummy_x)

# 只保存推理部分
tf.saved_model.save(model, "saved_model_best",
    signatures=model.call.get_concrete_function(
        tf.TensorSpec(shape=(None, IMG_SIZE[0], IMG_SIZE[1], 3), dtype=tf.float32)
    )
)

# 转换为 TFLite
converter = tf.lite.TFLiteConverter.from_saved_model("saved_model_best")
converter.optimizations = [tf.lite.Optimize.DEFAULT]  # 可选量化
tflite_model = converter.convert()

# 保存 .tflite 文件
with open("vit_model.tflite", "wb") as f:
    f.write(tflite_model)

print("✅ SavedModel & TFLite 已生成：saved_model_best / vit_model.tflite")
