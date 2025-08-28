import paho.mqtt.client as mqtt

MQTT_BROKER = "127.0.0.1"
MQTT_PORT = 1883
SUBSCRIBE_TOPIC = "grpc_sub/weights"

# MQTT 回调
def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("[MQTT] Connected successfully")
        client.subscribe(SUBSCRIBE_TOPIC)
        print(f"[MQTT] Subscribed to topic: {SUBSCRIBE_TOPIC}")
    else:
        print(f"[MQTT] Connection failed with code {rc}")

def on_message(client, userdata, msg):
    print(f"[MQTT] Message arrived! topic={msg.topic}")
    print(f"[MQTT] Payload: {msg.payload.decode()}")

# 创建客户端
client = mqtt.Client()
client.on_connect = on_connect
client.on_message = on_message
client.username_pw_set("tim", "tim")

# 连接并启动循环
client.connect(MQTT_BROKER, MQTT_PORT, 60)
client.loop_start()  # 后台线程处理消息

print("[SYSTEM] MQTT client running. Waiting for messages from ESP32...")

try:
    while True:
        pass  # 主线程空转
except KeyboardInterrupt:
    print("\n[SYSTEM] Exiting...")
    client.loop_stop()
    client.disconnect()
