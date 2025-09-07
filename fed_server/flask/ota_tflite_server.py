from flask import Flask, request, jsonify, send_file
import numpy as np
import tensorflow as tf
from io import BytesIO
from typing import Union

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))  # RL 根目录
from global_hyparm import *
from util_hvac_PPO import TensorFlowESP32BaseExporter

app = Flask(__name__)

# 全局保存 exporter 和 TFLite 模型
exporter = None
tflite_model_bytes = None
ota_metadata = None

@app.route("/init_exporter", methods=["POST"])
def init_exporter():
    global exporter
    data = request.json
    model_path = data.get("model_path")
    if not model_path:
        return jsonify({"error": "model_path required"}), 400

    policy_model = tf.keras.models.load_model(model_path)
    exporter = TensorFlowESP32BaseExporter(policy_model)
    return jsonify({"status": "exporter initialized"})

@app.route("/generate_ota_package", methods=["POST"])
def generate_ota_package():
    """
    生成 TFLite 模型和 OTA 元数据
    """
    global exporter, tflite_model_bytes, ota_metadata
    if exporter is None:
        return jsonify({"error": "exporter not initialized"}), 400

    num_samples = request.json.get("num_samples", 10)
    quantize = request.json.get("quantize", True)
    prune = request.json.get("prune", True)
    firmware_version = request.json.get("firmware_version", "1.0.0")

    # 模拟环境生成代表性数据
    class DummyEnv:
        def reset(self):
            return np.random.rand(exporter.model.input_shape[1]).astype(np.float32)

    rep_data = exporter.create_representative_dataset(DummyEnv(), num_samples)
    tflite_model_bytes = exporter.convert_to_tflite(rep_data, quantize=quantize, prune=prune)

    # 生成 OTA 元数据，不包含模型
    ota_metadata = {
        "firmware_version": firmware_version,
        "model_format": "tflite",
        "input_shape": list(exporter.model.input_shape[1:]),
        "output_shapes": [output.shape[1:].as_list() for output in exporter.model.outputs],
        "quantized": quantize,
        "pruned": prune,
        "file_size": len(tflite_model_bytes),
        "checksum": exporter._calculate_checksum(tflite_model_bytes)
    }

    return jsonify({"status": "OTA package generated", "metadata": ota_metadata})

@app.route("/download_tflite", methods=["GET"])
def download_tflite():
    """
    直接返回二进制 TFLite 文件
    """
    global tflite_model_bytes
    if tflite_model_bytes is None:
        return jsonify({"error": "TFLite model not generated yet"}), 400

    return send_file(BytesIO(tflite_model_bytes),
                     mimetype="application/octet-stream",
                     as_attachment=True,
                     download_name="model.tflite")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
