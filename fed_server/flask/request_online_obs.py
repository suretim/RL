
from flask_cors import CORS
import numpy as np
from flask import Flask, request, jsonify, send_file
import os
import hashlib
from collections import deque
import threading
import time
import random

import  queue
#set FLASK_APP=D:\RL\RL\fed_server\flask\ota_server.py
#set FLASK_ENV=development
#flask run --host=192.168.0.57 --port=5000

app = Flask(__name__)
CORS(app)



# 模擬環境數據生成
def simulate_env_data():
    while True:
        obs = [random.random() for _ in range(5)]  # 示例觀測
        qbuffer.push(obs)
        time.sleep(0.1)  # 每100ms一筆數據




class OTAServer:
    def __init__(self, ota_package_path, latest_version="1.0.0"):
        self.ota_package_path = ota_package_path
        with open(ota_package_path, 'rb') as f:
            self.package_data = f.read()
        self.package_hash = hashlib.md5(self.package_data).hexdigest()
        self.latest_version = latest_version  # 固定服务器最新版本

    def get_firmware_info(self, device_id, current_version):
        """检查设备是否需要更新"""
        update_needed = current_version != self.latest_version
        info = {
            'update_available': update_needed,
            'latest_version': self.latest_version,
            'package_size': len(self.package_data),
            'hash': self.package_hash
        }
        return info

@app.route('/upload', methods=['POST'])
def upload_client_data():
    data = request.json
    # data 可以包含 client_id, features, labels, model_params 等
    print("Received from client:", data.keys())
    return jsonify({'status':'ok'})




# 初始化 OTA 服务
ota_handler = OTAServer("saved_models/ppo_model.bin", latest_version="1.0.0")


@app.route('/api/check-update/<device_id>/<current_version>')
def check_update(device_id, current_version):
    return jsonify(ota_handler.get_firmware_info(device_id, current_version))


@app.route('/api/download-update')
def download_update():
    return send_file("esp32_ota_package.json", as_attachment=True)


@app.route("/api/bin-update")
def bin_update():

    info = ota_handler.get_firmware_info("dummy", "0.0.0")
    if info['update_available']:
        return send_file(ota_handler.ota_package_path, as_attachment=True)
    else:
        return send_file(ota_handler.ota_package_path, as_attachment=True)
        #return jsonify({"message": "No update available"}), 404

# 提供 OTA JSON
@app.route("/esp32_ota_package.json")
def serve_ota_package():
    return send_file("esp32_ota_package.json", mimetype="application/json")




# 模型存放路徑
MODEL_DIR = "../esp32_model"
ACTOR_MODEL = "actor.tflite"
CRITIC_MODEL = "critic.tflite"

# 計算文件 hash，方便 ESP32 檢查版本
def get_file_hash(file_path):
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()

@app.route("/ota/actor")
def download_actor():
    path = os.path.join(MODEL_DIR, ACTOR_MODEL)
    if not os.path.exists(path):
        return "Actor model not found", 404
    return send_file(path, mimetype="application/octet-stream")

@app.route("/ota/critic")
def download_critic():
    path = os.path.join(MODEL_DIR, CRITIC_MODEL)
    if not os.path.exists(path):
        return "Critic model not found", 404
    return send_file(path, mimetype="application/octet-stream")

@app.route("/ota/version")
def model_version():
    actor_path = os.path.join(MODEL_DIR, ACTOR_MODEL)
    critic_path = os.path.join(MODEL_DIR, CRITIC_MODEL)
    if not os.path.exists(actor_path) or not os.path.exists(critic_path):
        return jsonify({"error": "model not found"}), 404

    actor_hash = get_file_hash(actor_path)
    critic_hash = get_file_hash(critic_path)

    return jsonify({
        "actor_hash": actor_hash,
        "critic_hash": critic_hash
    })



# ========== QBuffer ==========
class QBuffer:
    def __init__(self, maxlen=1000):
        self.buffer = deque(maxlen=maxlen)
        self.lock = threading.Lock()

    def push(self, item):
        with self.lock:
            self.buffer.append(item)

    def get_seq(self, idx):
        with self.lock:
            if len(self.buffer) == 0:
                return None
            return self.buffer[idx % len(self.buffer)]

# 全局實例
qbuffer = QBuffer(maxlen=1000)
process_q = queue.Queue()
current_idx = 0

# ========== Flask ==========
app = Flask(__name__)

@app.route('/push_data', methods=['POST'])
def push_data():
    """
    外部設備推送觀測數據到服務器
    POST JSON 格式: {"obs": [0.1, 25.0, 0.5, 500, 600]}
    """
    data = request.get_json()
    if not data or "obs" not in data:
        return jsonify({"error": "invalid data"}), 400

    obs = data["obs"]
    if not isinstance(obs, list) or len(obs) != 5:
        return jsonify({"error": "invalid observation length"}), 400

    # 放到處理隊列，馬上返回
    process_q.put(obs)
    return jsonify({"status": "queued"})

@app.route('/seq_input', methods=['GET'])
def get_seq_input():
    global current_idx
    sample = qbuffer.get_seq(current_idx)
    current_idx += 1
    if sample is None:
        return jsonify({"error": "no data"}), 404
    return jsonify(sample)

# ========== 背景 Worker ==========
def worker():
    while True:
        obs = process_q.get()  # 阻塞等待
        try:
            qbuffer.push(obs)
            print(f"[worker] pushed: {obs}")
        except Exception as e:
            print(f"[worker error] {e}")
        process_q.task_done()

# ========== 主入口 ==========
if __name__ == "__main__":
    # 啟動 worker 線程
    threading.Thread(target=worker, daemon=True).start()
    # 注意: debug 模式請加 use_reloader=False 避免線程重複啟動
    app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False)

    os.makedirs(MODEL_DIR, exist_ok=True)

