import numpy as np
import tensorflow as tf
from tensorflow import keras
from pathlib import Path
import h5py
from flwr_pub import FLWR_PUB as publ
from flwr.client import NumPyClient
from utils import DataLoader

# 加载简单模型
class ESP32Client(NumPyClient):
    def __init__(self, device_id, data_dir, model,data_loader, augment=False):
        self.device_id = device_id
        self.data_dir = data_dir
        self.model = model
        self.data_loader = data_loader
        self.augment = augment

    def get_parameters(self, config):
        return self.model.get_weights()

    def set_parameters(self, parameters):
        self.model.set_weights(parameters)

    def fit(self, parameters, config):
        # 设置模型参数

        self.set_parameters(parameters)

        # 加载数据
        features, labels = data_loader.load_data()

        # 配置训练参数
        batch_size = config.get("batch_size", 32)
        epochs = config.get("epochs", 1)
        lr = config.get("lr", 0.001)

        # 创建数据集
        dataset = tf.data.Dataset.from_tensor_slices((features, labels))
        dataset = dataset.shuffle(len(features)).batch(batch_size)

        # 配置优化器
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()

        # 训练指标
        train_loss = tf.keras.metrics.Mean()
        train_acc = tf.keras.metrics.SparseCategoricalAccuracy()

        # 训练循环
        for epoch in range(epochs):
            for x_batch, y_batch in dataset:
                with tf.GradientTape() as tape:
                    preds = self.model(x_batch, training=True)
                    loss = loss_fn(y_batch, preds)
                    loss += sum(self.model.losses)  # 添加正则化损失

                grads = tape.gradient(loss, self.model.trainable_variables)
                optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

                # 更新指标
                train_loss(loss)
                train_acc(y_batch, preds)

            print(f"设备 {self.device_id} - Epoch {epoch + 1} - "
                  f"Loss: {train_loss.result():.4f} - "
                  f"Accuracy: {train_acc.result():.4f}")

        # 返回更新后的参数和指标
        # 获取所有可训练参数（按层顺序排列）
        #all_weights = model.get_weights()

        second_layer = model.layers[1]

        # 正确获取权重矩阵（kernel）
        second_layer_weights = second_layer.kernel.numpy()
        second_layer_bias = second_layer.kernel.numpy()
        print("权重矩阵形状:", second_layer_weights.shape)

        # 获取偏置（如果存在）
        if second_layer.use_bias:
            second_layer_bias = second_layer.bias.numpy()
            print("偏置向量形状:", second_layer_bias.shape)
        # 获取第二层的偏置
        print("第二层权重:", second_layer_weights)
        print("第二层偏置:", second_layer_bias)
        #updated_weights = self.model.get_weights()

        metrics = {
            "train_loss": float(train_loss.result()),
            "train_accuracy": float(train_acc.result()),
            "num_samples": len(features)
        }

        return second_layer_weights,second_layer_bias, len(features), metrics

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)

        # 使用训练数据作为验证（实际中应使用独立验证集）
        features, labels = self.data_loader.load_data()
        # 创建数据集
        batch_size = config.get("batch_size", 32)
        dataset = tf.data.Dataset.from_tensor_slices((features, labels))
        dataset = dataset.batch(batch_size)

        # 评估指标
        val_loss = tf.keras.metrics.Mean()
        val_acc = tf.keras.metrics.SparseCategoricalAccuracy()

        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()

        for x_batch, y_batch in dataset:
            preds = self.model(x_batch, training=False)
            loss = loss_fn(y_batch, preds)

            val_loss(loss)
            val_acc(y_batch, preds)

        metrics = {
            "val_loss": float(val_loss.result()),
            "val_accuracy": float(val_acc.result())
        }

        return float(val_loss.result()), len(features), metrics



# 加载简单模型
def create_model() -> tf.keras.Model:
    model = keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(64,)),
        tf.keras.layers.Dense(3, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

model = create_model()

# 获取初始参数
initial_params = [param.numpy() for param in  model.trainable_variables]
# 创建数据加载器
data_loader = DataLoader( data_dir="data",device_id="esp32_001" )

# 2. 创建客户端
client = ESP32Client( data_dir="data",device_id="esp32_001",model=model, data_loader=data_loader)
# 3. 模拟联邦学习训练轮次
#initial_params = model.get_weights()
config = {
    "batch_size": 64,
    "epochs": 3,
    "lr": 0.01
}

# 3. 获取参数
basic_params = client.get_parameters(config)
dense_weights,dense_bias, num_examples, metrics=client.fit(basic_params, config=config)

MQTT_BROKER = "192.168.0.57"
#MQTT_BROKER = "192.168.68.237"
#MQTT_BROKER = "127.0.0.1"
MQTT_PORT = 1883
FEDER_PUBLISH = "federated_model/parameters"
mq_pub=publ(device_id="esp32_001",
           mqtt_broker=MQTT_BROKER,
           mqtt_port=MQTT_PORT,
           topic=FEDER_PUBLISH
           )
mq_pub.mqtt_pub_metadata(dense_weights,dense_bias)
print(f"基础参数数量: {metrics}")

'''
# 获取压缩后的参数
compressed_params = client.get_parameters({
    "compress": "quantize",
    "quant_bits": 4
})

# 获取特定层的参数
selected_params = client.get_parameters({
    "only_trainable": True,
    "exclude_layers": ["batch_normalization"]
})

print(f"基础参数数量: {len(basic_params)}")
print(f"压缩后参数数量: {len(compressed_params)}")
'''