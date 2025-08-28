import tensorflow as tf
import numpy as np
#import matplotlib.pyplot as plt
#import os
from tensorflow.keras import layers, models
import json
import paho.mqtt.client as mqtt

import model_pb2
import model_pb2_grpc
#C:\Users\Administrator\miniconda3\Library\bin\conda.bat
#C:\Users\Administrator\miniconda3\Scripts\conda.exe
#C:\Users\Administrator\miniconda3\condabin\conda.bat
#conda activate my_env
#cd C:\tim\aicam\main\fed_server\cloud_models
#python emqx_manager.py
#netstat -ano | findstr :18083

#MQTT_BROKER = "192.168.0.57"
MQTT_BROKER = "127.0.0.1"
MQTT_PORT = 1883
FEDER_PUBLISH = "federated_model/parameters"
# Function: Convert some hex value into an array for C programming
def hex_to_c_array(hex_data, var_name):

    c_str = ''

    # Create header guard
    c_str += '#ifndef ' + var_name.upper() + '_H\n'
    c_str += '#define ' + var_name.upper() + '_H\n\n'

    # Declare C variable
    c_str += 'alignas(16) const unsigned char ' + var_name + '[] = {'
    hex_array = []
    for i, val in enumerate(hex_data):

        # Construct string from hex
        hex_str = format(val, '#04x')

        # Add formatting so each line stays within 80 characters
        if (i + 1) < len(hex_data):
            hex_str += ','
        if (i + 1) % 12 == 0:
            hex_str += '\n '
        hex_array.append(hex_str)

    # Add closing brace
    c_str += '\n ' + format(' '.join(hex_array)) + '\n};\n\n'

    # Add array length at top of file
    #c_str += '\nunsigned int ' + var_name + '_len = ' + str(
    #    len(hex_data)) + ';\n'

    c_str += '\nunsigned int ' + var_name + '_len = ' + 'sizeof(' + var_name + ');\n\n'

    # Close out header guard
    c_str += '#endif //' + var_name.upper() + '_H'

    return c_str

def build_encoderx (input_shape=(64, 64, 3)):
    model = models.Sequential([
        layers.Conv2D(16, 3, activation='relu', padding='same'),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, activation='relu', padding='same'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation='relu', padding='same'),
        #  替换 GlobalAveragePooling2D
        layers.AveragePooling2D(pool_size=(16, 16)),  # 假设最终尺寸是 4x4 → 手动设置 pool
        layers.Flatten(),                             # 输出 shape: (batch, 64)
    ])
    return model

# ====================
# 1. 构建轻量CNN特征提取器
# ====================
'''
def build_encoder (input_shape=(64, 64, 3)):
    model = tf.keras.Sequential([
        layers.Conv2D(16, (3, 3), activation='relu', input_shape=(64, 64, 3)),
        layers.MaxPooling2D(),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.AveragePooling2D(pool_size=(4, 4)),  # Replace mean operation
        layers.Flatten(),
        layers.Dense(64)
    ])
    return model
'''
encoder = build_encoderx()
#dummy_input = tf.random.normal((1, 64, 64, 3))  # CIFAR-10 的 shape
#_ = encoder(dummy_input)  # 執行一次 forward
#encoder.summary()

# ====================
# 2. 加载图片数据（需自备）
# ====================
# 假设你有两类图片在以下目录：
# data/
#   └── healthy/
#   └── disease/
img_size = (64, 64)

def load_dataset(data_dir):
    return tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        labels='inferred',
        label_mode='int',
        image_size=img_size,
        batch_size=32
    )

dataset = load_dataset("data")  # 替换为你的路径
class_names = dataset.class_names
print("Classes:", class_names)

# ====================
# 3. 编码 + 计算类中心（prototype）
# ====================
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
        proto = embeddings[labels == label].mean(axis=0)
        prototypes[label] = proto
    return prototypes
prototypes=compute_prototypes(encoder, dataset)
# ====================
# 4. 分类函数：计算距离
# ====================
def predict_prototype(encoder, image, prototypes):
    emb = encoder(tf.expand_dims(image, 0)).numpy()[0]
    distances = {k: np.linalg.norm(emb - v) for k, v in prototypes.items()}
    pred = min(distances, key=distances.get)
    return pred, distances

# ====================
# 5. 训练模型（用于特征提取）
# ====================
encoder.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

# 将 encoder + 分类头拼起来训练
model = tf.keras.Sequential([
    encoder,
    layers.Dense(len(class_names), activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(dataset, epochs=20)

encoder_model = model.layers[0]
#encoder_model.predict(np.zeros((64, 64, 3)), np.zeros((64, 64, 3)))
#predict_prototype(encoder_model, np.zeros((64, 64, 3)), prototypes)
sample_image = tf.random.normal([1, 64, 64, 3])  # 示例输入
embeddings = encoder_model.predict(sample_image)
print("Embeddings shape:", embeddings.shape)
prediction, distances = predict_prototype(encoder_model, sample_image[0], prototypes)
print("Prediction:", prediction)
c_model_name = 'encoder_model'

# ====================
# 6. 导出为 TensorFlow Lite 模型（部署用）
# ====================
# 去掉分类头，仅导出 encoder
#encoder.save(c_model_name+".h5")

#converter = tf.lite.TFLiteConverter.from_keras_model(encoder)
#converter.optimizations = [tf.lite.Optimize.DEFAULT]  # 8-bit量化
#tflite_model = converter.convert()

#import tensorflow as tf

#model = tf.keras.models.load_model('encoder_model.h5')

converter = tf.lite.TFLiteConverter.from_keras_model(encoder)

# ✅ 保证所有层都是 float32
converter.optimizations = []  # 不使用 INT8 量化
converter.target_spec.supported_types = [tf.float32]

# 不允许混合
converter.inference_input_type = tf.float32
converter.inference_output_type = tf.float32

tflite_model = converter.convert()

with open(c_model_name+".tflite", "wb") as f:
    f.write(tflite_model)

'''
# Write TFLite model to a C source (or header) file
with open(c_model_name + '.h', 'w') as file:
    file.write(hex_to_c_array(tflite_model, c_model_name))

# 获取所有的权重 tensor
# 加载 tflite 模型
interpreter = tf.lite.Interpreter(model_path="encoder_model.tflite")
interpreter.allocate_tensors()
tensors = interpreter.get_tensor_details()

weights = []
for tensor in tensors:
    if 'weight' in tensor['name'].lower() or 'kernel' in tensor['name'].lower():
        data = interpreter.get_tensor(tensor['index'])
        weights.append(data.flatten())  # 展平为 1D
# 合并所有参数为一个 list
if not weights:
    raise ValueError("No weights to concatenate. The list is empty.")
weights = [np.array(w) if not isinstance(w, np.ndarray) else w for w in weights]
flat_weights = np.concatenate(weights).tolist()

#flat_weights = np.concatenate(weights).tolist()
print(f"提取了 {len(flat_weights)} 个参数")
'''

# 获取分类头的权重和偏置
dense_layer = model.layers[-1]  # 获取最后一层（分类头）
weights, biases = dense_layer.get_weights()  # 权重矩阵和偏置向量


# 构建消息
        msg_weights = model_pb2.ModelParams()
        msg_weights.param_type = model_pb2.CLASSIFIER_WEIGHT
        msg_weights.values.extend(weights.flatten().tolist())
        msg_weights.client_id = client_id  # 可选设置 client_id
        payload_weights = msg_weights.SerializeToString()
        mqtt_client.publish(FEDER_PUBLISH, payload_weights)
        print(f"Published model parameters to MQTT: {payload_weights}")
        msg_bias = model_pb2.ModelParams()
        msg_bias.param_type = model_pb2.CLASSIFIER_BIAS
        msg_bias.values.extend(par2.flatten().tolist())
        msg_bias.client_id = client_id  # 可选设置 client_id
        payload_bias = msg_bias.SerializeToString()
        mqtt_client.publish(FEDER_PUBLISH, payload_bias)



# 转换为可JSON序列化的格式
weights_data = {
    "weights": weights.tolist(),  # 将numpy数组转为列表
    "biases": biases.tolist(),
    "metadata": {
        "num_classes": len(class_names),
        "input_shape": weights.shape[0]  # 输入特征维度
    }
}

# 打包为JSON字符串
json_payload = json.dumps(weights_data)
print("JSON Payload Size:", len(json_payload), "bytes")


mqtt_client = mqtt.Client()

def mqtt_pub_metadata():
    #client = mqtt.Client()
    #mqtt_client.on_connect = on_connect
    #mqtt_client.on_message = on_message
    # 设置用户名和密码
    username = "tim"  # 替换为你的 MQTT 用户名
    password = "tim"  # 替换为你的 MQTT 密码
    mqtt_client.username_pw_set(username, password)  # 设置用户名和密码
    # 设置重连超时时间，单位为毫秒
    reconnect_timeout_ms = 10000  # 10秒的重连超时
    mqtt_client.reconnect_delay_set(min_delay=1, max_delay=10)  # 设置重连延迟（最小1秒，最大10秒）

    mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 60)  # 更换为实际的 MQTT broker 地址
    #mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 60)
    #mqtt_client.loop_start()
    # 让程序持续运行，以便接收和处理消息
    #try:
    #    while True:
    #        time.sleep(1)  # 可以适当调整为更小的时间间隔，确保不中断 MQTT 事件处理
    #except KeyboardInterrupt:
    #    print("Disconnected from MQTT broker.")
    #    mqtt_client.loop_stop()

    #client.connect("192.168.0.57", 1883)

    #message = json.dumps({"weights": flat_weights})
    mqtt_client.publish(FEDER_PUBLISH, json_payload)
mqtt_pub_metadata()

# Write TFLite model to a C source (or header) file
#c_model_name = 'encoder_model'
#with open(c_model_name + '.h', 'w') as file:
#    file.write(hex_to_c_array(tflite_model, c_model_name))
