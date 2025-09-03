# ota_server.py

import flask
from flask import Flask, jsonify, send_file
import hashlib
import os
#python -c "import flask; print(flask.__version__)"
app = Flask(__name__)
ota_package_path=""

class OTAServer:
    def __init__(self, ota_package_path):
        self.ota_package_path = ota_package_path
        with open(ota_package_path, 'rb') as f:
            self.package_data = f.read()
        self.package_hash = hashlib.md5(self.package_data).hexdigest()

    def get_firmware_info(self, device_id, current_version):
        """检查设备是否需要更新"""
        return {
            'update_available': current_version != "1.0.0",
            'latest_version': "1.0.0",
            'package_size': len(self.package_data),
            'hash': self.package_hash
        }


ota_handler = OTAServer("esp32_model.ota")


@app.route('/api/check-update/<device_id>/<current_version>')
def check_update(device_id, current_version):
    return jsonify(ota_handler.get_firmware_info(device_id, current_version))


@app.route('/api/download-update')
def download_update():
    return send_file(ota_package_path, as_attachment=True)


@app.route('/')
def hello():
    return "Hello World!"

if __name__ == '__main__':
    app.run(debug=True)
    #app.run(host='0.0.0.0', port=5000)