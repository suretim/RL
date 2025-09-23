import tensorflow as tf
import numpy as np
import json
import os
from tf_utils import create_representative_dataset_from_ESP32BaseExporter
from util_agent import ESP32PPOAgent,ESP32OnlinePPOFisherAgent
from util_exporter import TensorFlowESP32Exporter


def train_and_export():
    # 1. 训练模型
    agent = ESP32OnlinePPOFisherAgent()
    agent.learn(total_timesteps=1000000)

    # 2. 保存原始模型
    #agent.policy.save("trained_policy_tf")

    # 3. 创建导出器
    exporter = TensorFlowESP32Exporter(agent.policy)

    # 4. 生成代表性数据
    representative_data = create_representative_dataset_from_ESP32BaseExporter(
        agent.policy, agent.env, num_samples=1000
    )

    # 5. 创建OTA包
    exporter.save_ota_base_package(
        "esp32_ota_package.json",
        representative_data,
        firmware_version="1.0.0",
        prune=True,
        quantize=True
    )


def create_representative_dataset_from_generator(policy, env, num_samples=1000):
    """从生成器创建代表性数据集"""
    exporter = TensorFlowESP32Exporter(policy)
    return exporter.create_representative_dataset(env, num_samples)

# cc62673 (flak and esp32 v.1.4)

# 使用示例
if __name__ == "__main__":

    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

    esp32_agent = ESP32PPOAgent(state_dim=5, action_dim=4, hidden_units=8)
    train_and_export()
    # 顯示模型信息
    print("模型參數數量:")
    print(f"Actor: {esp32_agent._count_params(esp32_agent.actor)}")
    print(f"Critic: {esp32_agent._count_params(esp32_agent.critic)}")

    # 導出ESP32所需文件
    esp32_agent.export_for_esp32()