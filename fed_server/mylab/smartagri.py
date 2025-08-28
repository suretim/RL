import paho.mqtt.client as mqtt
import json
import time
from datetime import datetime
import logging
import random
import sys

# ==== 配置日志 ====
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# ==== MQTT 参数配置 ====
MQTT_CONFIG = {
    "broker": "192.168.0.57",  # MQTT 代理地址
    "port": 1883,  # MQTT 端口
    "topic": "smartagriculture1/leaf_detection",  # 发布主题
    "username": "tim",  # 认证用户名
    "password": "tim",  # 认证密码
    "client_id": f"leaf_detector_{random.randint(1000, 9999)}",  # 客户端ID
    "keepalive": 60,  # 心跳间隔(秒)
    "qos": 1,  # 服务质量等级 (0, 1, 2)
    "retain": False  # 是否保留消息
}

# ==== 推理结果模拟数据 ====
TENT_ID = "TENT_A01"
PREDICTED_CLASS = "disease"  # or "healthy"
DISTANCE = 0.22


# ==== MQTT 回调函数 ====
def on_connect(client, userdata, flags, rc):
    """连接回调"""
    if rc == 0:
        logger.info("✅ Connected to MQTT Broker!")
    else:
        logger.error(f"❌ Connection failed with code {rc}")
        if rc == 5:
            logger.error("Authentication error. Check username/password")


def on_disconnect(client, userdata, rc):
    """断开连接回调"""
    if rc != 0:
        logger.warning(f"⚠️ Unexpected disconnection (rc={rc}). Reconnecting...")
        reconnect(client)


def on_publish(client, userdata, mid):
    """消息发布回调"""
    logger.info(f"📢 Message published (mid={mid})")


def reconnect(client, max_retries=5):
    """带重试的重新连接"""
    retry_count = 0
    while retry_count < max_retries:
        try:
            logger.info(f"↻ Attempting reconnect ({retry_count + 1}/{max_retries})...")
            client.reconnect()
            return
        except Exception as e:
            logger.error(f"Reconnect failed: {str(e)}")
            retry_count += 1
            time.sleep(2 ** retry_count)  # 指数退避

    logger.critical("🔥 Maximum reconnect attempts reached. Exiting.")
    sys.exit(1)


# ==== 主函数 ====
def main():
    # 创建MQTT客户端
    client = mqtt.Client(
        client_id=MQTT_CONFIG["client_id"],
        clean_session=True
    )

    # 设置回调函数
    client.on_connect = on_connect
    client.on_disconnect = on_disconnect
    client.on_publish = on_publish

    # 设置认证凭据
    if MQTT_CONFIG["username"] and MQTT_CONFIG["password"]:
        client.username_pw_set(
            MQTT_CONFIG["username"],
            MQTT_CONFIG["password"]
        )

    try:
        # 连接代理
        logger.info(f"🔗 Connecting to {MQTT_CONFIG['broker']}:{MQTT_CONFIG['port']}...")
        client.connect(
            MQTT_CONFIG["broker"],
            port=MQTT_CONFIG["port"],
            keepalive=MQTT_CONFIG["keepalive"]
        )
        client.loop_start()  # 启动后台线程处理消息

        # 构造消息载荷
        payload = {
            "tent_id": TENT_ID,
            "timestamp": datetime.utcnow().isoformat() + "Z",  # UTC时间带时区
            "prediction": PREDICTED_CLASS,
            "distance": round(DISTANCE, 3),
            "sensor_id": MQTT_CONFIG["client_id"]
        }

        # 发布消息
        result = client.publish(
            topic=MQTT_CONFIG["topic"],
            payload=json.dumps(payload),
            qos=MQTT_CONFIG["qos"],
            retain=MQTT_CONFIG["retain"]
        )

        # 等待消息发送完成
        result.wait_for_publish(timeout=5)
        if result.rc == mqtt.MQTT_ERR_SUCCESS:
            logger.info(f"📡 Sent MQTT message to [{MQTT_CONFIG['topic']}]:\n{json.dumps(payload, indent=2)}")
        else:
            logger.error(f"Failed to publish message: {mqtt.error_string(result.rc)}")

        # 短暂等待确保消息发送
        time.sleep(1)

    except Exception as e:
        logger.exception(f"🔥 Critical error: {str(e)}")
    finally:
        # 清理断开连接
        client.loop_stop()
        client.disconnect()
        logger.info("🔌 Disconnected from MQTT broker")


if __name__ == "__main__":
    main()