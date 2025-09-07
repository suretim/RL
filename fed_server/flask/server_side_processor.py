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
from util_hvac_PPO import ESP32PPOAgent,LifelongPPOBaseAgent
#from train_and_export import LifelongPPOAgent, TensorFlowESP32BaseExporter


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


# -----------------------------
# 使用範例（修正成可 load 的完整模型檔）
# -----------------------------
if __name__ == "__main__":
    agent = ESP32PPOWithFisher(state_dim=5, action_dim=4, hidden_units=8)

    # 假設已訓練好 actor
    representative_data = np.random.randn(100, 5).astype(np.float32)
    agent.compute_fisher_matrix(representative_data)

    # 保存 TFLite
    agent.save_tflite_model("esp32_actor.tflite" )
    #agent.actor.save("esp32ppo_actor.h5")
    # 保存 Fisher & Optimal Params
    agent.save_fisher_and_params("esp32_fisher.npz")
    agent=OnlineESP32PPOAgent()
    agent.actor.save("esp32ppo_actor.h5")

    exporter = TensorFlowESP32Exporter("esp32ppo_actor.h5")

    # 代表性資料（請替換成你的實際資料）
    representative_data = np.random.randn(1000, *exporter.model.input_shape[1:]).astype(np.float32)

    # 建立並保存 OTA
    exporter.save_ota_package(
        output_path="esp32_model_ota.json",
        representative_data=representative_data,
        firmware_version="1.0.0",
        prune=True,
        quantize=True
    )

    # 也可單獨呼叫
    #_ = exporter.apply_quantization(representative_data)
    #_ = exporter.compute_fisher_matrix(representative_data)
