# tf_utils.py
import tensorflow as tf
import numpy as np
from util_hvac_agent import ESP32OnlinePPOFisherAgent
from util_exporter import TensorFlowESP32Exporter



def create_representative_dataset_from_ESP32BaseExporter(policy, env, num_samples=1000):
    """从生成器创建代表性数据集"""
    agent=ESP32OnlinePPOFisherAgent()
    exporter = TensorFlowESP32Exporter(policy)

    return create_representative_dataset(env,agent.policy, num_samples)

def create_representative_dataset(self, env,  num_samples=1000, steps=10, policy_net=None):
    """创建代表性数据集，基于当前状态选择动作"""
    representative_data = []

    for _ in range(num_samples):
        obs = env.reset()

        # 运行环境，采集多步状态
        for _ in range(steps):
            # 根据当前的状态选择动作（假设你有一个训练好的策略网络）
            action = policy_net.predict(obs)  # 假设 `predict` 方法返回一个动作

            # 执行动作，获取新的状态
            obs, _, _, _ = env.step(action)

            if isinstance(obs, tuple):
                obs = obs[0]
            representative_data.append(obs.astype(np.float32))

    return np.array(representative_data)


def create_representative_dataset_from_modelpredict(model, env, num_samples=100):
    """
    从环境生成器创建代表性数据集
    """
    observations = []

    obs = env.reset()
    if isinstance(obs, tuple):
        obs = obs[0]

    for _ in range(num_samples):
        observations.append(obs)
        action = model.predict(obs.reshape(1, -1))[0]
        obs, _, done, _ = env.step(action)

        if done:
            obs = env.reset()
            if isinstance(obs, tuple):
                obs = obs[0]

    return np.array(observations)


def convert_to_tflite(model, output_path="converted_model.tflite"):
    """
    简单转换Keras模型到TFLite
    """
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    with open(output_path, 'wb') as f:
        f.write(tflite_model)

    return tflite_model


def analyze_tflite_model(model_path):
    """
    分析TFLite模型信息
    """
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print("Input details:", input_details)
    print("Output details:", output_details)

    return input_details, output_details