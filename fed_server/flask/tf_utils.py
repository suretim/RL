# tf_utils.py
import tensorflow as tf
import numpy as np

from util_hvac_PPO import TensorFlowESP32BaseExporter

def create_representative_dataset_from_ESP32BaseExporter(policy, env, num_samples=1000):
    """从生成器创建代表性数据集"""
    exporter = TensorFlowESP32BaseExporter(policy)
    return exporter.create_representative_dataset(env, num_samples)

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