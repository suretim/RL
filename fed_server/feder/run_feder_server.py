import paho.mqtt.client as mqtt
import time
import datetime
import threading

# MQTT 代理配置（根据实际情况修改）
MQTT_BROKER = "127.0.0.1"  # 公共测试服务器
MQTT_PORT = 1883
TOPIC = "capture/mqttx_"  # 替换为你的主题
CLIENT_ID = "python_publisher"

# 创建MQTT客户端
mqtt_client = mqtt.Client(client_id=CLIENT_ID)

username = "tim"  # 替换为你的 MQTT 用户名
password = "tim"  # 替换为你的 MQTT 密码
mqtt_client.username_pw_set(username, password)  # 设置用户名和密码
# 设置重连超时时间，单位为毫秒
reconnect_timeout_ms = 10000  # 10秒的重连超时
mqtt_client.reconnect_delay_set(min_delay=1, max_delay=10)  # 设置重连延迟（最小1秒，最大10秒）

def on_connect(client, userdata, flags, rc):
    """连接回调函数"""
    if rc == 0:
        print(f"成功连接到MQTT代理 @ {MQTT_BROKER}")
    else:
        print(f"连接失败，错误码: {rc}")


def publish_message():
    """每分钟发布消息的定时任务"""
    while True:
        # 生成带时间戳的消息
        #message = f"定时消息 @ {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        message = f"weight/prams"

        # 发布消息
        result = mqtt_client.publish(TOPIC, message, qos=1)

        # 检查发布状态
        if result.rc == mqtt.MQTT_ERR_SUCCESS:
            print(f"已发布: {message} → [{TOPIC}]")
        else:
            print(f"发布失败，错误码: {result.rc}")

        # 等待60秒
        time.sleep(60)


# 设置回调函数
mqtt_client.on_connect = on_connect

try:
    # 连接到代理
    mqtt_client.connect(MQTT_BROKER, MQTT_PORT, keepalive=60)

    # 启动网络循环线程
    mqtt_client.loop_start()

    # 创建定时发布线程
    publish_thread = threading.Thread(target=publish_message)
    publish_thread.daemon = True  # 设为守护线程
    publish_thread.start()

    # 主线程保持运行
    while True:
        time.sleep(1)

except KeyboardInterrupt:
    print("\n程序终止")
except Exception as e:
    print(f"发生错误: {str(e)}")
finally:
    mqtt_client.disconnect()
    mqtt_client.loop_stop()
    print("MQTT连接已关闭")