import numpy as np
import tensorflow as tf
from tensorflow import keras
import flwr as fl
from flwr.server.strategy import FedAvg
from flwr.server import ServerConfig  # 导入 ServerConfig
# 加载简单模型
def create_model() -> tf.keras.Model:
    model = keras.Sequential([
        keras.layers.Conv2D(32, 3, activation='relu', input_shape=(64, 64, 3)),
        keras.layers.MaxPooling2D(),
        keras.layers.Conv2D(64, 3, activation='relu'),
        keras.layers.MaxPooling2D(),
        keras.layers.GlobalAveragePooling2D(),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# 创建评估函数
def get_evaluation_fn(model: tf.keras.Model):
    def evaluate(weights: np.ndarray):
        model.set_weights(weights)
        loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
        return loss, {"accuracy": accuracy}
    return evaluate

# Flower 策略
strategy = FedAvg(
    evaluate_fn=get_evaluation_fn(create_model()),
    fraction_fit=0.5,
    min_fit_clients=3,
    min_available_clients=3,
)

# 使用 ServerConfig 创建配置
server_config = ServerConfig(
    num_rounds=3  # 设置训练轮数
)

# 启动 Flower 服务器
def start_server():
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        strategy=strategy,  # 使用策略来配置
        config=server_config,  # 传递 ServerConfig 配置
    )

if __name__ == "__main__":
    start_server()
