from flask import Flask, request, jsonify
from queue import Queue
import threading
import time
import json
import paho.mqtt.client as mqtt

# ========== Flask ==========
app = Flask(__name__)

process_q = Queue()
img_health = 0.0  # 健康指标全局变量

# ========== MQTT ==========
MQTT_BROKER = "localhost"   # 改成你的 MQTT 服务器 IP
MQTT_PORT = 1883
MQTT_TOPIC = "growhealth/obs"  # ESP32 上传数据的主题

def on_connect(client, userdata, flags, rc):
    print(f"[MQTT] Connected with result code {rc}")
    client.subscribe(MQTT_TOPIC)

def on_message(client, userdata, msg):
    global img_health
    try:
        payload = json.loads(msg.payload.decode())
        obs = payload.get("obs", [])
        if isinstance(obs, list) and len(obs) == 5:
            process_q.put(obs)
            print(f"[MQTT] Received obs: {obs}")
        else:
            print("[MQTT] Invalid obs format")
    except Exception as e:
        print(f"[MQTT] Error: {e}")

# 启动 MQTT 客户端线程
def mqtt_thread():
    client = mqtt.Client()
    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(MQTT_BROKER, MQTT_PORT, 60)
    client.loop_forever()

threading.Thread(target=mqtt_thread, daemon=True).start()

# ========== 数据处理后台线程 ==========
def background_worker():
    global img_health
    while True:
        obs = process_q.get()
        # 示例：用平均值计算植物健康度
        img_health = round(sum(obs) / len(obs), 2)
        print(f"[Worker] Updated img_health = {img_health}")
        process_q.task_done()
        time.sleep(0.1)

threading.Thread(target=background_worker, daemon=True).start()

# ========== Flask 路由 ==========
@app.route('/push_data', methods=['POST'])
def push_data():
    """允许通过 HTTP 推送观测数据（备用方式）"""
    global img_health
    data = request.get_json()
    if not data or "obs" not in data:
        return jsonify({"error": "invalid data"}), 400
    obs = data["obs"]
    if not isinstance(obs, list) or len(obs) != 5:
        return jsonify({"error": "invalid observation length"}), 400

    process_q.put(obs)
    return jsonify({"status": img_health})

@app.route('/status', methods=['GET'])
def get_status():
    """获取当前植物健康状态"""
    global img_health
    return jsonify({"health": img_health})

if __name__ == "__main__":
    print("[System] Grow img health Server Running...")
    app.run(host="0.0.0.0", port=5003)
