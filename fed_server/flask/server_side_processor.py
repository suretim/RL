# server_side_processor_tf.py
import tensorflow as tf
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


class ESP32PPOWithFisher(ESP32PPOAgent):
    def __init__(self, state_dim=5, action_dim=4, hidden_units=8, ewc_lambda=500):
        super().__init__(state_dim, action_dim, hidden_units)
        self.ewc_lambda = ewc_lambda
        self.fisher_matrix = None
        self.optimal_params = None

    def compute_fisher_matrix(self, dataset: np.ndarray):
        """計算 Fisher 矩陣"""
        fisher = {}
        optimal_params = {}
        for var in self.actor.trainable_variables:
            fisher[var.name] = np.zeros_like(var.numpy())
            optimal_params[var.name] = var.numpy().copy()

        for x in dataset:
            x = x[None, ...].astype(np.float32)
            with tf.GradientTape() as tape:
                probs = self.actor(x)
                log_prob = tf.math.log(probs + 1e-8)
            grads = tape.gradient(log_prob, self.actor.trainable_variables)
            for g, var in zip(grads, self.actor.trainable_variables):
                if g is not None:
                    fisher[var.name] += (g.numpy() ** 2) / len(dataset)

        self.fisher_matrix = fisher
        self.optimal_params = optimal_params

    def save_fisher_and_params(self, path: str):
        np.savez_compressed(
            path,
            fisher={k: v for k,v in self.fisher_matrix.items()},
            optimal={k: v for k,v in self.optimal_params.items()}
        )
        print(f"✅ Fisher matrix & optimal params saved to {path}")

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

class OnlineESP32PPOAgent(ESP32PPOAgent):
    """專為ESP32設計的輕量級PPO代理，支持 online EWC 和 TFLite 導出"""

    def __init__(self, state_dim=5, action_dim=4,
                 clip_epsilon=0.2, value_coef=0.5, entropy_coef=0.01,
                 ewc_lambda=500, memory_size=1000, hidden_units=8):
        super().__init__(state_dim, action_dim, clip_epsilon, value_coef,
                         entropy_coef, ewc_lambda, memory_size)

        self.hidden_units = hidden_units
        self._tflite_models = {}

        self.actor = self._build_esp32_actor()
        self.critic = self._build_esp32_critic()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

        # EWC online
        self.online_fisher = None
        self.optimal_params = None
        self.ema_decay = 0.99
        self.fisher_update_frequency = 1
        self.update_counter = 0

        print(f"ESP32代理初始化完成: {hidden_units}隱藏單元")

    def _build_esp32_actor(self):
        return tf.keras.Sequential([
            tf.keras.layers.Dense(self.hidden_units, activation='relu', input_shape=(self.state_dim,)),
            tf.keras.layers.Dense(self.action_dim, activation='sigmoid')
        ], name='esp32_actor')

    def _build_esp32_critic(self):
        return tf.keras.Sequential([
            tf.keras.layers.Dense(self.hidden_units, activation='relu', input_shape=(self.state_dim,)),
            tf.keras.layers.Dense(1, activation='linear')
        ], name='esp32_critic')

    # ------------------ 新增 policy 方法 ------------------
    def policy(self, obs):
        """
        返回動作分佈 (Bernoulli) 供 PPO loss 計算
        obs: tf.Tensor, shape (batch_size, state_dim)
        """
        logits = self.actor(obs)
        return tfd.Bernoulli(probs=logits)

    # ------------------ EWC online 更新 ------------------
    def update_online_fisher(self, obs: tf.Tensor, action: tf.Tensor):
        self.update_counter += 1
        if self.update_counter % self.fisher_update_frequency != 0:
            return

        if self.online_fisher is None:
            self._initialize_online_fisher()

        with tf.GradientTape() as tape:
            dist = self.policy(obs)
            log_prob = dist.log_prob(action)
            log_prob_mean = tf.reduce_mean(log_prob)

        grads = tape.gradient(log_prob_mean, self.actor.trainable_variables)
        for var, grad in zip(self.actor.trainable_variables, grads):
            if grad is not None:
                var_name = var.name
                grad_sq = tf.square(grad)
                old_fisher = self.online_fisher.get(var_name, tf.zeros_like(var))
                self.online_fisher[var_name] = self.ema_decay * old_fisher + (1 - self.ema_decay) * grad_sq

    def _initialize_online_fisher(self):
        self.online_fisher = {}
        for var in self.actor.trainable_variables:
            self.online_fisher[var.name] = tf.zeros_like(var)

    def ewc_regularization(self, current_params):
        if self.online_fisher is None or self.optimal_params is None:
            return tf.constant(0.0, dtype=tf.float32)

        ewc_loss = tf.constant(0.0, dtype=tf.float32)
        for var in self.actor.trainable_variables:
            name = var.name
            fisher = self.online_fisher.get(name)
            optimal = self.optimal_params.get(name)
            current = current_params.get(name)
            if fisher is not None and optimal is not None and current is not None:
                ewc_loss += tf.reduce_sum(fisher * tf.square(current - optimal))
        return self.ewc_lambda * ewc_loss

    # ------------------ PPO update with online EWC ------------------
    def update_with_online_ewc(self, obs_batch, action_batch, advantage_batch, old_log_prob_batch):
        current_params = {var.name: var for var in self.actor.trainable_variables}

        with tf.GradientTape() as tape:
            dist = self.policy(obs_batch)
            new_log_prob = dist.log_prob(action_batch)
            ratio = tf.exp(new_log_prob - old_log_prob_batch)
            clipped_ratio = tf.clip_by_value(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
            policy_loss = -tf.reduce_mean(tf.minimum(ratio * advantage_batch,
                                                     clipped_ratio * advantage_batch))
            value_loss = tf.reduce_mean(tf.square(self.critic(obs_batch) - advantage_batch))
            entropy_loss = -tf.reduce_mean(dist.entropy())
            ewc_loss = self.ewc_regularization(current_params)
            total_loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss + ewc_loss

        grads = tape.gradient(total_loss, self.actor.trainable_variables + self.critic.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.actor.trainable_variables + self.critic.trainable_variables))

        # 更新 online Fisher
        for i in range(obs_batch.shape[0]):
            self.update_online_fisher(obs_batch[i:i+1], action_batch[i:i+1])

        return {
            'total_loss': total_loss.numpy(),
            'policy_loss': policy_loss.numpy(),
            'value_loss': value_loss.numpy(),
            'entropy_loss': entropy_loss.numpy(),
            'ewc_loss': ewc_loss.numpy()
        }


# -----------------------------
# 獨立的 TensorFlow → ESP32 匯出器（不再繼承 Agent）
# -----------------------------
class TensorFlowESP32Exporter:
    def __init__(self, model_or_path: Union[str, tf.keras.Model]):
        """
        Args:
            model_or_path: 已訓練 Keras 模型或可由 tf.keras.models.load_model 載入的路徑
                           **注意**: 不支援僅權重檔（*.weights.h5）。請改用 .keras 或 SavedModel。
        """
        if isinstance(model_or_path, str):
            # FIX: 只允許可 load 的完整模型
            self.model = tf.keras.models.load_model(model_or_path)
        else:
            self.model = model_or_path

        self.quantized_model_bytes: Optional[bytes] = None
        self.fisher_matrix: Optional[Dict[str, np.ndarray]] = None
        self.optimal_params: Optional[Dict[str, np.ndarray]] = None

    # 量化
    def apply_quantization(self, representative_data: np.ndarray) -> bytes:
        """
        將模型量化為 int8，適合 ESP32。
        """
        print("Applying post-training quantization...")

        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

        # 定義 generator 函數
        def representative_dataset_gen():
            for i in range(min(100, len(representative_data))):
                # 每次返回一個 batch (1, input_shape)
                yield [representative_data[i:i + 1].astype(np.float32)]

        # 注意這裡要傳函數，不要傳 generator 對象
        converter.representative_dataset = representative_dataset_gen

        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8

        tflite_model = converter.convert()

        # 保存到文件示例
        with open('quantized_model.tflite', 'wb') as f:
            f.write(tflite_model)

        print("Quantization completed! Model saved as quantized_model.tflite")
        return tflite_model


    def _representative_dataset_gen(self, dataset: np.ndarray):
        # 注意：這裡是函數，yield 產生數據
        for i in range(min(100, len(dataset))):
            yield [dataset[i:i + 1].astype(np.float32)]


    def prune_model(self, target_sparsity: float = 0.5) -> tf.keras.Model:
        print(f"Pruning model to {target_sparsity * 100}% sparsity...")

        # 確保模型已編譯
        if not hasattr(self.model, 'loss') or self.model.loss is None:
            self.model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                loss='mse',
                metrics=['mae']
            )

        # 剪枝代碼
        pruning_params = {
            'pruning_schedule': tfmot.sparsity.keras.ConstantSparsity(
                target_sparsity, begin_step=0, frequency=100
            )
        }

        pruned_model = tfmot.sparsity.keras.prune_low_magnitude(
            self.model, **pruning_params
        )

        pruned_model.compile(
            optimizer=self.model.optimizer,
            loss=self.model.loss,
            metrics=self.model.metrics
        )

        # 微調可以在這裡進行
        # pruned_model.fit(x_train, y_train, epochs=2, validation_data=(x_val, y_val))

        self.model = tfmot.sparsity.keras.strip_pruning(pruned_model)
        print("Pruning completed!")
        return self.model

    # 剪枝

    # Fisher（對分類輸出安全處理）
    def compute_fisher_matrix(self, dataset: np.ndarray, num_samples: int = 1000) -> Dict[str, np.ndarray]:
        print("Computing Fisher Information Matrix...")
        vars_list = self.model.trainable_variables
        self.optimal_params = {v.name: v.numpy().copy() for v in vars_list}
        fisher = {v.name: np.zeros_like(v.numpy()) for v in vars_list}

        @tf.function
        def grads_on_input(x):
            with tf.GradientTape() as tape:
                y = self.model(x, training=False)
                # 若輸出不是機率，轉 logits 為機率（避免 log(0)）
                if y.dtype != tf.float32:
                    y = tf.cast(y, tf.float32)
                # 小偏移避免 log(0)
                logp = tf.math.log(tf.nn.softmax(y) + 1e-8)
                # 對 batch 取均值 → 純量
                obj = tf.reduce_mean(tf.reduce_sum(logp, axis=-1))
            return tape.gradient(obj, vars_list)

        n = min(num_samples, len(dataset))
        for i in range(n):
            if i % 100 == 0:
                print(f"Processing sample {i}/{n}")
            x = tf.convert_to_tensor(dataset[i:i+1], dtype=tf.float32)
            grads = grads_on_input(x)
            for v, g in zip(vars_list, grads):
                if g is not None:
                    fisher[v.name] += (g.numpy() ** 2) / float(n)

        self.fisher_matrix = fisher
        print("Fisher matrix computation completed!")
        return fisher

    # 壓縮
    def compress_for_esp32(self, tflite_bytes: bytes) -> Dict[str, Any]:
        print("Compressing model for ESP32...")
        comp = zlib.compress(tflite_bytes)
        original_size = len(tflite_bytes)
        compressed_size = len(comp)
        compression_ratio = original_size / compressed_size if compressed_size > 0 else float('inf')
        print(f"Compression ratio: {compression_ratio:.2f}x")
        print(f"Original: {original_size} bytes, Compressed: {compressed_size} bytes")
        return {
            'compressed_model': comp,
            'original_size': original_size,
            'compressed_size': compressed_size,
            'compression_ratio': compression_ratio,  # FIX: 供 metadata 使用
        }

    # OTA 組包
    def create_ota_package(self,
                           representative_data: np.ndarray,
                           firmware_version: str = "1.0.0",
                           prune: bool = True,
                           quantize: bool = True) -> Dict[str, Any]:
        print("Creating OTA package...")
        if prune:
            self.prune_model(target_sparsity=0.5)

        self.compute_fisher_matrix(representative_data)

        if quantize:
            tflite_bytes = self.apply_quantization(representative_data)
        else:
            converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
            tflite_bytes = converter.convert()

        comp = self.compress_for_esp32(tflite_bytes)
        simplified_fisher = self._simplify_fisher_matrix()
        metadata = {
            'firmware_version': firmware_version,
            'model_type': 'TFLite',
            'input_shape': list(self.model.input_shape[1:]) if self.model.input_shape else None,
            'output_shape': list(self.model.output_shape[1:]) if self.model.output_shape else None,
            'compression_ratio': comp['compression_ratio'],
            'timestamp': datetime.now().isoformat(),
            'quantization': quantize,
            'pruning': prune,
        }

        # FIX: JSON 不能存 bytes，先 Base64
        model_data_b64 = base64.b64encode(comp['compressed_model']).decode('ascii')

        ota_package = {
            'metadata': metadata,
            'model_data_b64': model_data_b64,
            'fisher_matrix': simplified_fisher,
            'optimal_params': self._prepare_optimal_params(),
            'crc32': self._calculate_crc(comp['compressed_model']),
        }
        return ota_package

    def _simplify_fisher_matrix(self) -> Dict[str, Any]:
        simplified = {}
        assert self.fisher_matrix is not None
        for name, fm in self.fisher_matrix.items():
            thr = np.percentile(fm, 80.0)
            mask = fm > thr
            values = fm[mask]
            indices = np.where(mask)
            simplified[name] = {
                'values': values.astype(np.float32).tolist(),
                'indices': [idx.astype(np.int32).tolist() for idx in indices],
                'shape': list(fm.shape),
                'threshold': float(thr),
            }
        return simplified

    def _prepare_optimal_params(self) -> Dict[str, Any]:
        assert self.optimal_params is not None
        return {name: arr.astype(np.float32).tolist() for name, arr in self.optimal_params.items()}

    def _calculate_crc(self, data: bytes) -> int:
        return zlib.crc32(data)

    def save_ota_package(self, output_path: str, representative_data: np.ndarray, **kwargs) -> None:
        ota = self.create_ota_package(representative_data, **kwargs)

        # JSON 檔（Base64 模型）
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(ota, f, indent=2, ensure_ascii=False)

        # 二進位（含 bytes）— 使用 pickle
        binary_path = output_path.replace('.json', '.bin')
        import pickle
        with open(binary_path, 'wb') as f:
            pickle.dump(ota, f)

        print(f"OTA package saved to {output_path} and {binary_path}")
        print(f"JSON size: {os.path.getsize(output_path)} bytes")
        print(f"Binary size: {os.path.getsize(binary_path)} bytes")


from tensorflow import keras

class ESP32PPOAgent:
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
