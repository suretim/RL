import tensorflow as tf
import numpy as np
#from util_hvac_agent import LifelongPPOBaseAgent
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


class ESP32BaseExporter:
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


# -----------------------------
# 獨立的 TensorFlow → ESP32 匯出器（不再繼承 Agent）
# -----------------------------
class TensorFlowESP32Exporter(ESP32BaseExporter):
    def __init__(self, model_or_path: Union[str, tf.keras.Model], quantize: bool = False):
        super().__init__(model_or_path)
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
        self.ota_package=None
        self.quantize = quantize

    def convert_model_to_tflite(self) -> bytes:
        """
        Converts the model to TFLite without applying quantization.
        """
        print("Converting model to TFLite without quantization...")
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        converter.optimizations = []  # No optimizations (no quantization)
        tflite_model = converter.convert()
        return tflite_model
    # 量化
    def apply_quantization(self, representative_data: np.ndarray) -> bytes:
        """
        將模型量化為 int8,適合 ESP32。
        """
        if not self.quantize:
            print("Quantization is disabled. Returning the original model.")
            # Return the original model if quantization is disabled
            return self.convert_model_to_tflite()
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



    def prune_model(self, target_sparsity=0.5, fine_tune_data=None, epochs=2, batch_size=32):

        prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude
        # 剪枝參數 (多項式衰減)
        pruning_params = {
            "pruning_schedule": tfmot.sparsity.keras.PolynomialDecay(
                initial_sparsity=0.0,
                final_sparsity=target_sparsity,
                begin_step=0,
                end_step=1000
            )
        }

        # 包裝模型，加上 pruning wrapper
        pruned_model = prune_low_magnitude(self.model, **pruning_params)

        # 編譯
        pruned_model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )

        if fine_tune_data is not None:
            if isinstance(fine_tune_data, tuple) and len(fine_tune_data) == 2:
                # 有 (x, y) → 正常 supervised 微調
                x_train, y_train = fine_tune_data
                # 去掉多余维度
                if x_train.ndim == 3 and x_train.shape[1] == 1:
                    x_train = np.squeeze(x_train, axis=1)  # [batch, 1, feature_dim] → [batch, feature_dim]

                pruned_model.compile(
                    optimizer="adam",
                    loss="sparse_categorical_crossentropy",
                    metrics=["accuracy"]
                )
                callbacks = [
                    tfmot.sparsity.keras.UpdatePruningStep(),  # 剪枝必须回调
                    tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)  # 可选
                ]
                pruned_model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs,callbacks=callbacks, verbose=1)
            else:
                # 只有 x → 偽微調 (dummy label)
                x_train = fine_tune_data
                # 去掉多余维度
                if x_train.ndim == 3 and x_train.shape[1] == 1:
                    x_train = np.squeeze(x_train, axis=1)  # [batch, 1, feature_dim] → [batch, feature_dim]

                dummy_y = np.zeros((x_train.shape[0],), dtype=np.int32)  # fake label
                pruned_model.compile(
                    optimizer="adam",
                    loss="sparse_categorical_crossentropy"
                )
                pruned_model.fit(x_train, dummy_y, epochs=1)  # 跑短暫 1 epoch 就行
                print("⚠️ Warning: Fake fine-tune only, model may lose accuracy.")
        else:
            print("⚠️ No fine-tune data provided. Pruning without adaptation may hurt accuracy.")



        # strip_pruning 真正移除 wrapper
        final_model = tfmot.sparsity.keras.strip_pruning(pruned_model)
        return final_model

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
    def compress_for_esp32(self, tflite_bytes, compress=False):  # 添加compress参数
        """
        为ESP32压缩模型数据

        Args:
            tflite_bytes: TFLite模型字节数据
            compress: 是否进行压缩，默认为True
        """
        if compress:
            # 压缩逻辑
            compressed = zlib.compress(tflite_bytes)
            compression_ratio = len(tflite_bytes) / len(compressed)
            return {
                'compressed_model': compressed,
                'compression_ratio': compression_ratio,
                'compressed': True
            }
        else:
            # 不压缩，直接返回原始数据
            return {
                'compressed_model': tflite_bytes,
                'compression_ratio': 1.0,
                'compressed': False
            }
    def create_ota_package(self,
                           representative_data: np.ndarray,
                           firmware_version: str = "1.0.0",
                           fine_tune_data=None,  #fine_tune_data=(x_train, y_train)
                           prune: bool = True,
                           compress=False,
                           quantize: bool = False) -> Dict[str, Any]:
        print("Creating OTA package...")
        if prune:
            if fine_tune_data is not None:
                self.prune_model( target_sparsity=0.5, fine_tune_data=fine_tune_data, epochs=2)
            else:
                super().prune_model(target_sparsity=0.5)

        self.compute_fisher_matrix(representative_data)

        if quantize:
            tflite_bytes = self.apply_quantization(representative_data)
        else:
            converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
            converter.optimizations = []
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
            converter.inference_input_type = tf.float32  # 确保输入类型为 float32
            converter.inference_output_type = tf.float32  # 确保输出类型为 float32
            tflite_bytes = converter.convert()
            print("converter.optimizations=[]")
            for layer in self.model.layers:
                print(f"Layer {layer.name}: {layer.__class__}")

        comp = self.compress_for_esp32(tflite_bytes, compress=compress)
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

    def save_ota_package(self, output_path: str, representative_data=None,fine_tune_data=None, quantize=False, **kwargs) -> None:
        self.ota_package = self.create_ota_package(representative_data=representative_data,
                                                   fine_tune_data=fine_tune_data,
                                                   quantize= quantize, **kwargs)

        # JSON 檔（Base64 模型）
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.ota_package, f, indent=2, ensure_ascii=False)

        # 二進位（含 bytes）— 使用 pickle
        binary_path = output_path.replace('.json', '.bin')
        import pickle
        with open(binary_path, 'wb') as f:
            pickle.dump(self.ota_package, f)

        print(f"OTA package saved to {output_path} and {binary_path}")
        print(f"JSON size: {os.path.getsize(output_path)} bytes")
        print(f"Binary size: {os.path.getsize(binary_path)} bytes")

