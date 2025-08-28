import paho.mqtt.client as mqtt
import json
import time
from datetime import datetime

# ==== 推理后的结果示例 ====
tent_id = "TENT_A01"
predicted_class = "disease"  # or "healthy"
distance = 0.22  # 可选，用于置信度参考

# ==== MQTT 参数配置 ====
MQTT_BROKER = "127.0.0.1"      # 改为你自己的 broker 地址
MQTT_PORT = 1883
#FEDER_SUBSCRIPT = "federated_model/parameters"
FEDER_PUBLISH = "federated_model/parameters"

# 创建 MQTT 客户端并连接
client = mqtt.Client()
client.connect(MQTT_BROKER, MQTT_PORT, 60)

# 构造上传的 JSON 消息
payload = {
    "tent_id": tent_id,
    "timestamp": datetime.now().isoformat(),
    "prediction": predicted_class,
    "distance": round(distance, 3)
}

# 发送消息
client.publish(FEDER_PUBLISH, json.dumps(payload))
print(f"📡 Sent MQTT message to [{FEDER_PUBLISH}]:\n{json.dumps(payload, indent=2)}")

client.disconnect()
