import tensorflow as tf
import numpy as np
from util_hvac_agent import LifelongPPOBaseAgent
import tensorflow_probability as tfp
import tensorflow_model_optimization as tfmot
import zlib
import base64
import json
import os

from typing import Union
from typing import Dict
from typing import *
from datetime import datetime


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


# -----------------------------
# 獨立的 TensorFlow → ESP32 匯出器（不再繼承 Agent）
# -----------------------------
class TensorFlowESP32Exporter:
    def __init__(self, model_or_path: Union[str, tf.keras.Model]):
        """
        Args:
            model_or_path: 已訓練 Keras 模型或可由 tf.keras.models.load_model 載入的路徑
                           **注意**: 不支援僅權重檔（*.weights.h5)。請改用 .keras 或 SavedModel
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
        將模型量化為 int8,適合 ESP32。
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



    def compute_fisher_matrix(self, dataset: np.ndarray, num_samples: int = 1000) -> Dict[str, np.ndarray]:
        print("Computing Fisher Information Matrix...")
        trainable_vars = self.model.trainable_variables
        self.optimal_params = {v.name: v.numpy().copy() for v in trainable_vars}
        fisher = {v.name: np.zeros_like(v.numpy()) for v in trainable_vars}

        # 预定义梯度计算函数
        @tf.function
        def compute_sample_gradients(x):
            with tf.GradientTape() as tape:
                predictions = self.model(x, training=False)
                predictions = tf.cast(predictions, tf.float32)

                # 使用稳定的softmax和log计算
                stable_pred = predictions - tf.reduce_max(predictions, axis=-1, keepdims=True)
                exp_pred = tf.exp(stable_pred)
                probs = exp_pred / tf.reduce_sum(exp_pred, axis=-1, keepdims=True)
                log_probs = tf.math.log(probs + 1e-12)

                # 目标函数：平均对数概率
                objective = tf.reduce_mean(tf.reduce_sum(log_probs, axis=-1))

            # 计算相对于所有可训练变量的梯度
            return tape.gradient(objective, trainable_vars)

        n = min(num_samples, len(dataset))
        for i in range(n):
            if i % 100 == 0:
                print(f"Processing sample {i}/{n}")

            # 准备输入数据
            sample = dataset[i]
            if hasattr(sample, 'shape'):
                # 重塑输入以匹配模型期望
                if len(sample.shape) == 1:
                    sample = sample.reshape(1, -1)
                elif len(sample.shape) == 3:
                    sample = sample.reshape(sample.shape[0], sample.shape[2])

            x = tf.convert_to_tensor(sample, dtype=tf.float32)

            # 计算梯度
            gradients = compute_sample_gradients(x)

            # 更新Fisher矩阵
            for var, grad in zip(trainable_vars, gradients):
                if grad is not None:
                    fisher[var.name] += (grad.numpy() ** 2) / n

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

