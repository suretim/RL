# server_side_processor_tf.py
import tensorflow as tf
from tensorflow import keras
import numpy as np
import zlib
import json
import os
import base64
from datetime import datetime
import tensorflow_model_optimization as tfmot
from typing import Dict, Any, Optional, Union

# 你專案內的類（保持不動，假設存在）
from util_exporter import TensorFlowESP32Exporter
from util_hvac_agent import ESP32PPOWithFisherAgent,OnlineESP32PPOAgent



def ewc_update(actor, fisher, optimal_params, current_params, learning_rate=1e-3, ewc_lambda=500):
    """
    使用 Fisher 矩陣更新 Actor 參數
    actor: tf.keras.Sequential
    fisher: dict
    optimal_params: dict
    current_params: dict, 可用 actor.trainable_variables
    """
    with tf.GradientTape() as tape:
        loss = 0.0
        for var in actor.trainable_variables:
            name = var.name
            diff = var - optimal_params[name]
            loss += tf.reduce_sum(fisher[name] * diff ** 2)
        loss *= ewc_lambda

    grads = tape.gradient(loss, actor.trainable_variables)
    for var, g in zip(actor.trainable_variables, grads):
        if g is not None:
            var.assign_sub(learning_rate * g)

import tensorflow_probability as tfp
import os

tfd = tfp.distributions


class xESP32PPOAgent:
    """專為ESP32設計的輕量級PPO代理"""

    def __init__(self, state_dim=5, action_dim=4,
                 clip_epsilon=0.2, value_coef=0.5, entropy_coef=0.01):
        # 這裡簡單建一個 policy (MLP)
        inputs = keras.Input(shape=(state_dim,))
        x = keras.layers.Dense(32, activation="relu")(inputs)
        outputs = keras.layers.Dense(action_dim, activation="softmax")(x)
        self.policy = keras.Model(inputs, outputs)

        # 其它 PPO 超參數
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef

    def act(self, state):
        """輸入 state -> 輸出 action"""
        state = tf.convert_to_tensor([state], dtype=tf.float32)
        probs = self.policy(state)
        action = tf.random.categorical(tf.math.log(probs), num_samples=1)
        return int(action[0, 0])

    def save_model(self, path="trained_policy_tf.keras"):
        """保存 policy 模型"""
        self.policy.save(path)
        print(f"✅ 模型已保存到 {path}")

    def load_model(self, path="trained_policy_tf.keras"):
        """載入 policy 模型"""
        self.policy = keras.models.load_model(path)
        print(f"✅ 模型已從 {path} 載入")


def generate_smart_representative_data(env, num_samples=1000, mode_weights=None):
    """
    智能生成代表性数据，覆盖不同模式和工况

    Args:
        env: PlantLLLHVACEnv实例
        num_samples: 总样本数
        mode_weights: 不同模式的权重 [growing, flowering, seeding]
    """
    if mode_weights is None:
        mode_weights = [0.4, 0.3, 0.3]  # 默认权重

    all_data = []

    # 为每个模式生成数据
    modes = ["growing", "flowering", "seeding"]

    for mode, weight in zip(modes, mode_weights):
        num_mode_samples = int(num_samples * weight)
        print(f"为模式 '{mode}' 生成 {num_mode_samples} 个样本")

        # 设置环境模式
        env.mode = mode
        env.reset()

        mode_data = []

        # 生成该模式的数据
        for i in range(num_mode_samples):
            # 更智能的动作选择（基于当前模式）
            if mode == "growing":
                # 生长模式：更注重温湿度控制
                action = np.random.choice([0, 1], size=4, p=[0.3, 0.7])
            elif mode == "flowering":
                # 开花模式：需要更多光照和CO2
                action = np.random.choice([0, 1], size=4, p=[0.5, 0.5])
            else:  # seeding
                # 结果模式：更稳定的环境
                action = np.random.choice([0, 1], size=4, p=[0.7, 0.3])

            # 执行动作
            true_label = modes.index(mode)  # 使用模式作为真实标签
            next_state, reward, done, info = env.step(action, true_label=true_label)

            # 添加数据点
            current_data_point = env.current_sequence[0, -1]
            mode_data.append(current_data_point)

            if done:
                env.reset()

        all_data.extend(mode_data)

    # 随机打乱数据
    all_data = np.array(all_data)
    np.random.shuffle(all_data)

    return all_data[:num_samples].astype(np.float32)


# -----------------------------
# 使用範例（修正成可 load 的完整模型檔）
# -----------------------------
if __name__ == "__main__":
    # 代表性資料（請替換成你的實際資料）
    # 创建环境
    from util_env import PlantLLLHVACEnv

    env = PlantLLLHVACEnv(seq_len=10, n_features=5, mode="growing")

    # 生成代表性数据
    representative_data = generate_smart_representative_data(env, num_samples=1000)

    # 确保形状匹配（如果需要序列数据）
    if len(representative_data.shape) == 2:
        representative_data = representative_data.reshape(-1, 1, 5)
    agent = ESP32PPOWithFisherAgent(state_dim=5, action_dim=4, hidden_units=8)

    # 假設已訓練好 actor
    #representative_data = np.random.randn(100, 5).astype(np.float32)
    agent.compute_fisher_matrix(representative_data)

    # 保存 TFLite
    agent.save_tflite_model("esp32_actor.tflite" )
    agent.actor.save("esp32ppo_actor.h5")
    # 保存 Fisher & Optimal Params
    agent.save_fisher_and_params("esp32_fisher.npz")


    policy_agent=OnlineESP32PPOAgent()
    policy_agent.actor.save("esp32_policy.h5")

    # 4. 创建导出器并生成OTA包
    exporter = TensorFlowESP32Exporter("esp32_policy.h5")

    # 5. 生成并保存OTA包
    exporter.save_ota_package(
        output_path="esp32_policy.json",
        representative_data=representative_data,
        firmware_version="1.0.0",
        prune=True,  # 启用剪枝
        quantize=True  # 启用量化
    )
    ota_package = exporter.create_ota_package(representative_data, quantize=True)
    compressed_bytes = base64.b64decode(ota_package['model_data_b64'])
    decompressed_bytes = zlib.decompress(compressed_bytes)

    with open("esp32_optimized_model.tflite", 'wb') as f:
        f.write(decompressed_bytes)

    # 也可單獨呼叫
    #_ = exporter.apply_quantization(representative_data)
    #_ = exporter.compute_fisher_matrix(representative_data)
