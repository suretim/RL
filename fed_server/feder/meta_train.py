import tensorflow as tf
import numpy as np
#import matplotlib.pyplot as plt
#import os
from tensorflow.keras import layers, models
import json
#import paho.mqtt.client as mqtt

#import model_pb2
#import model_pb2_grpc
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
class Meta_Train:
    def __init__(self,dataset_dir=None ):
        #self.model_parameters_list = []
        #self.model_labels_list = []
        self.dataset_dir = dataset_dir
        self.model_parameters_list = np.empty((0, 64))
        self.model_labels_list = np.empty((0,))
        self.client_id=None

    def build_encoderx (self,input_shape=(64, 64, 3)):
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
        model.compile(optimizer='adam',
                        loss='sparse_categorical_crossentropy',
                        metrics=['accuracy'])
        return model


    def load_dataset(self,dataset_dir,img_size):
        return tf.keras.preprocessing.image_dataset_from_directory(
            directory=dataset_dir,
            labels='inferred',
            label_mode='int',
            image_size=img_size,
            batch_size=32
        )

    def compute_prototypes(self,encoder, dataset):
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

    def predict_prototype(self,encoder, image, prototypes):
        emb = encoder(tf.expand_dims(image, 0)).numpy()[0]
        distances = {k: np.linalg.norm(emb - v) for k, v in prototypes.items()}
        pred = min(distances, key=distances.get)
        return pred, distances

    def build_encoder_dense(self,encoder):
        # 将 encoder + 分类头拼起来训练
        model = tf.keras.Sequential([
            encoder,
            layers.Dense(len(class_names), activation='softmax')
        ])
        model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        return model

    def save2tflite(self,encoder):
        converter = tf.lite.TFLiteConverter.from_keras_model(encoder)

        # ✅ 保证所有层都是 float32
        converter.optimizations = []  # 不使用 INT8 量化
        converter.target_spec.supported_types = [tf.float32]

        # 不允许混合
        converter.inference_input_type = tf.float32
        converter.inference_output_type = tf.float32

        tflite_model = converter.convert()

        with open(c_model_name + ".tflite", "wb") as f:
            f.write(tflite_model)

def mqtt_server_init(mqtt_broker,mqtt_port,username,password):
    #client = mqtt.Client()
    #mqtt_client.on_connect = on_connect
    #mqtt_client.on_message = on_message

    mqtt_client.username_pw_set(username, password)  # 设置用户名和密码
    # 设置重连超时时间，单位为毫秒
    reconnect_timeout_ms = 10000  # 10秒的重连超时
    mqtt_client.reconnect_delay_set(min_delay=1, max_delay=10)  # 设置重连延迟（最小1秒，最大10秒）

    mqtt_client.connect(mqtt_broker, mqtt_port, 60)  # 更换为实际的 MQTT broker 地址
    #mqtt_client.connect(mqtt_broker, mqtt_port, 60)
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
    #mqtt_client.publish(FEDER_PUBLISH, json_payload)

MQTT_BROKER = "127.0.0.1"
GRPC_SERVER = "127.0.0.1:50051"
MQTT_PORT = 1883
FEDER_PUBLISH = "federated_model/parameters"
MSG_PUBLISH = "msg/mqttx_"  # 替换为你的主题
#define MQTT_TOPIC_SUB "capture/mqttx_"

GRPC_SUBSCRIBE = "grpc_sub/weights"
if __name__ == '__main__':
    c_model_name = 'encoder_model'

    #data_dir = "../../../../data"
    dataset_dir = "../../../../dataset/data3"
    client_id=1
    mqtt_broker=MQTT_BROKER
    mqtt_port=MQTT_PORT
    # 设置用户名和密码
    username = "tim"  # 替换为你的 MQTT 用户名
    password = "tim"  # 替换为你的 MQTT 密码
    #mqtt_client = mqtt.Client()
    #mqtt_server_init(mqtt_broker=mqtt_broker,mqtt_port=mqtt_port,username=username,password=password)
    Meta_Train=Meta_Train(dataset_dir=dataset_dir )
    encoder = Meta_Train.build_encoderx()
    # dummy_input = tf.random.normal((1, 64, 64, 3))  # CIFAR-10 的 shape
    # _ = encoder(dummy_input)  # 執行一次 forward
    # encoder.summary()
    # 假设你有3类图片在以下目录：
    # data/
    #   └── h/
    #   └── d/
    #   └── w/
    img_size = (64, 64)

    dataset = Meta_Train.load_dataset(dataset_dir=dataset_dir,img_size=img_size)
    class_names = dataset.class_names
    print("Classes:", class_names)

    # ====================
    # 3. 编码 + 计算类中心（prototype）
    # ====================
    prototypes = Meta_Train.compute_prototypes(encoder=encoder, dataset=dataset)


    model_encoder_dense = Meta_Train.build_encoder_dense(encoder=encoder)

    model_encoder_dense.fit(dataset, epochs=20)
    model_encoder = model_encoder_dense.layers[0]
    Meta_Train.save2tflite(encoder=model_encoder)
    # encoder_model.predict(np.zeros((64, 64, 3)), np.zeros((64, 64, 3)))
    # predict_prototype(encoder_model, np.zeros((64, 64, 3)), prototypes)
    #sample_image = tf.random.normal([1, 64, 64, 3])  # 示例输入
    #embeddings = model_encoder.predict(sample_image)
    #print("Embeddings shape:", embeddings.shape)
    #prediction, distances = Meta_Train.predict_prototype(model_encoder, sample_image[0], prototypes)
    #print("Prediction:", prediction)


    if False:
        # 获取分类头的权重和偏置
        dense_layer = model_encoder_dense.layers[-1]  # 获取最后一层（分类头）
        weights, biases = dense_layer.get_weights()  # 权重矩阵和偏置向量
        # 构建消息
        msg_weights = model_pb2.ModelParams()
        msg_weights.param_type = model_pb2.ENCODER_WEIGHT
        msg_weights.values.extend(weights.flatten().tolist())
        msg_weights.client_id = client_id  # 可选设置 client_id
        payload_weights = msg_weights.SerializeToString()
        mqtt_client.publish(FEDER_PUBLISH, payload_weights)
        print(f"Published model parameters to MQTT: {payload_weights}")
        msg_bias = model_pb2.ModelParams()
        msg_bias.param_type = model_pb2.ENCODER_BIAS
        msg_bias.values.extend(biases.flatten().tolist())
        msg_bias.client_id = client_id  # 可选设置 client_id
        payload_bias = msg_bias.SerializeToString()
        mqtt_client.publish(FEDER_PUBLISH, payload_bias)
