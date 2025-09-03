import tensorflow as tf
import numpy as np
import json
import os
from util_hvac_PPO import *


def train_and_export():
    # 1. 训练模型
    agent = LifelongPPOAgent()
    agent.learn(total_timesteps=1000000)

    # 2. 保存原始模型
    agent.save("trained_policy_tf")

    # 3. 创建导出器
    exporter = TensorFlowESP32BaseExporter(agent.policy)

    # 4. 生成代表性数据
    representative_data = create_representative_dataset_from_generator(
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


# 假设的基类
class LifelongPPOAgent(LifelongPPOBaseAgent):
    def __init__(self, state_dim=5, action_dim=4):
        super().__init__( state_dim=5, action_dim=4)
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.policy = self._build_policy_network()
        self.env = self._create_env()

    def _build_policy_network(self):
        """构建策略网络"""
        inputs = tf.keras.layers.Input(shape=(self.state_dim,))
        x = tf.keras.layers.Dense(64, activation='relu')(inputs)
        x = tf.keras.layers.Dense(64, activation='relu')(x)

        # 动作分布参数 - 使用Dense层
        mean = tf.keras.layers.Dense(self.action_dim)(x)
        log_std = tf.keras.layers.Dense(self.action_dim)(x)  # 改为Dense层

        return tf.keras.Model(inputs=inputs, outputs=[mean, log_std])

    def _create_env(self):
        """创建模拟环境"""

        class MockEnv:
            def __init__(self, state_dim=5, action_dim=4):
                self.state_dim = state_dim
                self.action_dim = action_dim
                self.state = np.zeros(state_dim, dtype=np.float32)

            def reset(self):
                return np.random.randn(self.state_dim).astype(np.float32)

            def step(self, action):
                # Your step implementation
                next_state = np.random.randn(self.state_dim).astype(np.float32)
                reward = float(np.random.rand())
                done = np.random.rand() > 0.95
                return next_state, reward, done, {}

        return MockEnv()

    def learn(self, total_timesteps=1000000):
        """模拟训练过程"""
        print(f"Training for {total_timesteps} timesteps...")
        # 这里简化训练过程
        for i in range(total_timesteps // 1000):
            if i % 100 == 0:
                print(f"Step {i * 1000}/{total_timesteps}")

    def save(self, path):
        """保存模型"""
        self.policy.save_weights(path + '.weights.h5')
        print(f"Saved model weights to {path}.weights.h5")


class TensorFlowESP32BaseExporter:
    def __init__(self, policy_model=None):
        self.model = policy_model
        self.converter = None

    def create_representative_dataset(self, env, num_samples=1000):
        """创建代表性数据集"""
        representative_data = []
        for _ in range(num_samples):
            obs = env.reset()
            if isinstance(obs, tuple):
                obs = obs[0]
            representative_data.append(obs.astype(np.float32))
        return np.array(representative_data)

    def convert_to_tflite(self, representative_data=None, quantize=True, prune=True):
        """转换为TFLite模型"""
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)

        if quantize:
            # 设置量化参数
            converter.optimizations = [tf.lite.Optimize.DEFAULT]

            if representative_data is not None:
                def representative_dataset():
                    for data in representative_data:
                        yield [data.reshape(1, -1)]

                converter.representative_dataset = representative_dataset
                converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
                converter.inference_input_type = tf.int8
                converter.inference_output_type = tf.int8

        if prune:
            # 启用剪枝（如果适用）
            pass

        tflite_model = converter.convert()
        return tflite_model

    def save_ota_base_package(self, output_path, representative_data=None,
                         firmware_version="1.0.0", prune=True, quantize=True):
        """创建OTA包"""
        # 转换模型
        tflite_model = self.convert_to_tflite(representative_data, quantize, prune)

        # 保存TFLite模型
        tflite_path = output_path.replace('.json', '.tflite')
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)
        print(f"Saved TFLite model to {tflite_path}")

        # 创建OTA元数据
        ota_metadata = {
            "firmware_version": firmware_version,
            "model_format": "tflite",
            "input_shape": list(self.model.input_shape[1:]),
            "output_shapes": [output.shape[1:].as_list() for output in self.model.outputs],  # Convert each to list
            "quantized": quantize,
            "pruned": prune,
            "file_size": len(tflite_model),
            "checksum": self._calculate_checksum(tflite_model)
        }



        # 保存OTA包
        with open(output_path, 'w') as f:
            json.dump(ota_metadata, f, indent=2)

        print(f"Saved OTA package to {output_path}")
        print(f"OTA Metadata: {json.dumps(ota_metadata, indent=2)}")

    def _calculate_checksum(self, data):
        """计算简单的校验和"""
        return hash(data) % 1000000


def create_representative_dataset_from_generator(policy, env, num_samples=1000):
    """从生成器创建代表性数据集"""
    exporter = TensorFlowESP32BaseExporter(policy)
    return exporter.create_representative_dataset(env, num_samples)


# 使用示例
if __name__ == "__main__":
    train_and_export()