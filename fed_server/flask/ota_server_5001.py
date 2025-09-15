from flask import Flask, send_file,jsonify,request
import socket
import psutil
import os

MODEL_DIR = "./models"
app = Flask(__name__)

def get_local_ip():
    """自动获取本机的局域网 IP"""
    for iface, addrs in psutil.net_if_addrs().items():
        for addr in addrs:
            if addr.family == socket.AF_INET:
                ip = addr.address
                # 过滤掉 127.x.x.x 和 169.254.x.x
                if ip.startswith("127.") or ip.startswith("169.254."):
                    continue
                # 只取常见内网段
                if ip.startswith("192.168.") or ip.startswith("10.") or ip.startswith("172."):
                    return ip
    return "127.0.0.1"

ALLOWED_EXTENSIONS = {'json'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/api/model", methods=["GET"])
def get_model():
    # 从查询参数获取模型名称
    model_name = request.args.get("name", None)
    if not model_name:
        return jsonify({"error": "Model name is required"}), 400

    # 拼接文件路径
    model_file = os.path.join(MODEL_DIR, f"{model_name}.json")

    # 检查文件是否存在
    if not os.path.exists(model_file):
        return jsonify({"error": "Model file not found"}), 404

    # 返回文件
    return send_file(
        model_file,
        mimetype="application/json",
        as_attachment=False
    )


@app.route("/ota_model")
def ota_model():
    return send_file("./ppo_model.bin", as_attachment=True)

@app.route("/ota_model_md5")
def ota_model_md5(): 
    import hashlib
    try:
        with open("./ppo_model.bin", "rb") as f:
            file_data = f.read()
        md5 = hashlib.md5(file_data).hexdigest()
        print(f"md5: {md5}")
        return jsonify({"md5": md5, "status": "success"})
    except FileNotFoundError:
        return jsonify({"error": "Model file not found", "status": "error"}), 404
    except Exception as e:
        return jsonify({"error": str(e), "status": "error"}), 500

if __name__ == "__main__":
    ip = get_local_ip()
    port = 5001


    print(f" 你的局域网 IP 是: {ip}")
    print(f" ESP32 OTA URL : http://{ip}:{port}/ota_model")
    print(f" ESP32 MD5 URL : http://{ip}:{port}/ota_model_md5")
    print(f"GET http://{ip}:{port}/api/model?name=esp32_policy")
    print(f"spiffs esp32_optimized_model.tflite")

    app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)
    # 启动 Flask



