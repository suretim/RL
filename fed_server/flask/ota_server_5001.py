from flask import Flask, send_file,jsonify
import socket
import psutil

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
    print(f"💡 你的局域网 IP 是: {ip}")
    print(f"👉 ESP32 OTA URL 示例: http://{ip}:{port}/ota_model")
    print(f"👉 ESP32 MD5 URL 示例: http://{ip}:{port}/ota_model_md5")

    # 启动 Flask
    app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)
