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
from util_hvac_agent import ESP32OnlinePPOFisherAgent,ESP32PPOFisherAgent
from util_env import PlantLLLHVACEnv

MODEL_DIR = "./models"

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

def generate_smart_representative_data(env, num_samples=1000, mode_weights=None, return_labels=False):
    """
    智能生成代表性数据，覆盖不同模式和工况

    Args:
        env: PlantLLLHVACEnv实例
        num_samples: 总样本数
        mode_weights: 不同模式的权重 [growing, flowering, seeding]
        return_labels: 是否同时返回标签 (X, y)
    """
    if mode_weights is None:
        mode_weights = [0.4, 0.3, 0.3]  # 默认权重

    all_data = []
    all_labels = []

    # 为每个模式生成数据
    modes = ["growing", "flowering", "seeding"]

    for mode, weight in zip(modes, mode_weights):
        num_mode_samples = int(num_samples * weight)
        print(f"为模式 '{mode}' 生成 {num_mode_samples} 个样本")

        # 设置环境模式
        env.mode = mode
        env.reset()

        # 生成该模式的数据
        for i in range(num_mode_samples):
            if mode == "growing":
                action = np.random.choice([0, 1], size=4, p=[0.3, 0.7])
            elif mode == "flowering":
                action = np.random.choice([0, 1], size=4, p=[0.5, 0.5])
            else:  # seeding
                action = np.random.choice([0, 1], size=4, p=[0.7, 0.3])

            # 执行动作
            true_label = modes.index(mode)
            next_state, reward, done, info = env.step(action, true_label=true_label)

            # 添加数据点 & 标签
            current_data_point = env.current_sequence[0, -1]
            all_data.append(current_data_point)
            all_labels.append(true_label)

            if done:
                env.reset()

    # 转 numpy
    all_data = np.array(all_data, dtype=np.float32)
    all_labels = np.array(all_labels, dtype=np.int32)

    # 随机打乱
    idx = np.arange(len(all_data))
    np.random.shuffle(idx)
    all_data, all_labels = all_data[idx], all_labels[idx]

    if return_labels:
        return all_data[:num_samples], all_labels[:num_samples]
    else:
        return all_data[:num_samples]
from util_trainer import LLLTrainer

def env_pipe_trainer(lll_model=None,num_tasks=3,latent_dim=64,num_classes=3,num_epochs_per_task=3,batch_size=32, learning_rate=0.001, ewc_lambda=0.4):
    #env_lll_model, state_dim, action_dim, hidden_units,learning_rate=0.001, ewc_lambda=0.4

    trainer = LLLTrainer(lll_model=lll_model, learning_rate=0.001, ewc_lambda=0.4)

    # 模拟多任务数据
    for task_id in range(num_tasks):
        print(f"\n=== Training Task {task_id + 1} ===")
        num_samples = 200
        latent_features = np.random.randn(num_samples, latent_dim).astype(np.float32)
        labels = np.random.randint(0, num_classes, size=(num_samples,)).astype(np.int32)
        loss=0
        # 按 batch 训练当前任务
        for epoch in range(num_epochs_per_task):
            indices = np.random.permutation(num_samples)
            latent_features_shuffled = latent_features[indices]
            labels_shuffled = labels[indices]

            for start_idx in range(0, num_samples, batch_size):
                end_idx = start_idx + batch_size
                batch_latent = latent_features_shuffled[start_idx:end_idx]
                batch_labels = labels_shuffled[start_idx:end_idx]
                loss = trainer._train_lll_model(batch_latent, batch_labels)
            print(f"  Epoch {epoch + 1}, Last batch loss: {loss:.4f}")

        # 训练完当前任务后，更新 EWC 信息
        trainer.update_ewc(latent_features, labels)


# -----------------------------
# 使用範例（修正成可 load 的完整模型檔）
# -----------------------------
if __name__ == "__main__":
    # 代表性資料（請替換成你的實際資料）
    # 创建环境
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    latent_dim = 64
    action_dim=4
    num_classes = 3
    batch_size = 32
    num_epochs_per_task = 3
    num_tasks = 3
    env = PlantLLLHVACEnv(seq_len=10,   mode="growing")
    env_pipe_trainer(
        lll_model=env.lll_model ,
        num_tasks=num_tasks,
        latent_dim=latent_dim,
        num_classes=num_classes,
        num_epochs_per_task= num_epochs_per_task,
        batch_size=batch_size,
        learning_rate=0.001,
        ewc_lambda=0.4)

    # 生成代表性数据
    #representative_data = generate_smart_representative_data(env, num_samples=1000)
    representative_data, y_train    = generate_smart_representative_data(env, 100, return_labels=True)

    # 确保形状匹配（如果需要序列数据）
    if len(representative_data.shape) == 2:
        representative_data = representative_data.reshape(-1, 1, 5)
    agent = ESP32PPOFisherAgent(state_dim=env.n_features, action_dim=action_dim, hidden_units=8)

    agent.compute_fisher_matrix(representative_data)
    # 保存 Fisher & Optimal Params
    path_npz=os.path.join(MODEL_DIR, "esp32_fisher.npz")
    agent.save_fisher_and_params(path_npz)
    # 保存 TFLite
    #agent.save_tflite_model("esp32_actor.tflite" )
    path_h5 = os.path.join(MODEL_DIR, "esp32ppo_actor.h5")
    agent.actor.save(path_h5)



    policy_agent=ESP32OnlinePPOFisherAgent(fisher_matrix=agent.fisher_matrix,optimal_params=agent.optimal_params)
    path_policy_h5 = os.path.join(MODEL_DIR, "esp32_policy.h5")
    policy_agent.actor.save(path_policy_h5)
    policy_agent.actor.summary()
    # 4. 创建导出器并生成OTA包
    exporter = TensorFlowESP32Exporter(path_policy_h5)
    path_policy_json = os.path.join(MODEL_DIR, "esp32_policy.json")
    # 5. 生成并保存OTA包
    exporter.save_ota_package(
        output_path=path_policy_json,
        representative_data=representative_data,
        fine_tune_data=(representative_data, y_train),
        firmware_version="1.0.0",
        prune=True,  # 启用剪枝
        quantize=True  # 启用量化
    )
    ota_package = exporter.create_ota_package(representative_data, quantize=True)
    compressed_bytes = base64.b64decode(ota_package['model_data_b64'])
    decompressed_bytes = zlib.decompress(compressed_bytes)
    path_policy_tflite = os.path.join(MODEL_DIR, "esp32_optimized_model.tflite")
    with open(path_policy_tflite, 'wb') as f:
        f.write(decompressed_bytes)

    # 也可單獨呼叫
    #_ = exporter.apply_quantization(representative_data)
    #_ = exporter.compute_fisher_matrix(representative_data)
