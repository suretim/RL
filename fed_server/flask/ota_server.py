import hashlib
import os
from flask import Flask, jsonify, send_file, request
from flask_cors import CORS
import numpy as np
app = Flask(__name__)

#set FLASK_APP=D:\RL\RL\fed_server\flask\ota_server.py
#set FLASK_ENV=development
#flask run --host=192.168.0.57 --port=5000


CORS(app)

# 模拟全局序列数据 (seq_len=20, n_features=3)
GLOBAL_SEQ = np.random.rand(100, 20, 3).tolist()  # 100 个样本

# 当前分发索引
current_idx = 0

@app.route('/seq_input', methods=['GET'])
def get_seq_input():
    global current_idx
    t = request.args.get('t', default=0, type=int)

    # 循环分发
    sample = GLOBAL_SEQ[current_idx % len(GLOBAL_SEQ)]
    current_idx += 1

    return jsonify(sample)

@app.route('/upload', methods=['POST'])
def upload_client_data():
    data = request.json
    # data 可以包含 client_id, features, labels, model_params 等
    print("Received from client:", data.keys())
    return jsonify({'status':'ok'})



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


# 初始化 OTA 服务
ota_handler = OTAServer("ppo_model.bin", latest_version="1.0.0")


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

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
