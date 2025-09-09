import tensorflow as tf
import numpy as np
import json
import os
from tf_utils import create_representative_dataset_from_ESP32BaseExporter
from util_hvac_PPO import LifelongESP32PPOAgent,LifelongPPOBaseAgent,TensorFlowESP32BaseExporter


def train_and_export():
    # 1. 训练模型
    agent = LifelongESP32PPOAgent()
    agent.learn(total_timesteps=1000000)

    # 2. 保存原始模型
    agent.save("trained_policy_tf")

    # 3. 创建导出器
    exporter = TensorFlowESP32BaseExporter(agent.policy)

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
    exporter = TensorFlowESP32BaseExporter(policy)
    return exporter.create_representative_dataset(env, num_samples)

# cc62673 (flak and esp32 v.1.4)

# 使用示例
if __name__ == "__main__":
    train_and_export()