import numpy as np
import paho.mqtt.client as mqtt
import model_pb2
import json

# 模拟你训练得到的权重
# 构造假数据
#dense_weights = np.random.rand(64, 3).astype(np.float32)
#dense_bias = np.random.rand(3, 1).astype(np.float32)

class FLWR_PUB:
    def __init__(self,device_id,mqtt_broker,mqtt_port,topic ):
        self.device_id = device_id
        self.mqtt_broker=mqtt_broker
        self.mqtt_port = mqtt_port
        self.topic = topic

    def _set_msg(self,dense_weights,dense_bias):
        # 构建消息
        msg_weights = model_pb2.ModelParams()
        msg_weights.param_type = model_pb2.CLASSIFIER_WEIGHT
        msg_weights.values.extend(dense_weights.flatten().tolist())
        msg_weights.client_id = 1  # 可选设置 client_id
        payload_weights = msg_weights.SerializeToString()
        msg_bias = model_pb2.ModelParams()
        msg_bias.param_type = model_pb2.CLASSIFIER_BIAS
        msg_bias.values.extend(dense_bias.flatten().tolist())
        msg_bias.client_id = 2  # 可选设置 client_id
        # 序列化消息
        payload_bias    = msg_bias.SerializeToString()
        #json_payload = json.dumps(payload)
        return payload_weights, payload_bias

    def mqtt_pub_metadata(self,dense_weights,dense_bias):
        mqtt_client = mqtt.Client()
        #mqtt_client.on_connect = on_connect
        #mqtt_client.on_message = on_message
        # 设置用户名和密码
        username = "tim"  # 替换为你的 MQTT 用户名
        password = "tim"  # 替换为你的 MQTT 密码
        mqtt_client.username_pw_set(username, password)  # 设置用户名和密码
        # 设置重连超时时间，单位为毫秒
        reconnect_timeout_ms = 10000  # 10秒的重连超时
        mqtt_client.reconnect_delay_set(min_delay=1, max_delay=10)  # 设置重连延迟（最小1秒，最大10秒）

        mqtt_client.connect(self.mqtt_broker, self.mqtt_port, 60)  # 更换为实际的 MQTT broker 地址
        payload_weights,payload_bias=self._set_msg(dense_weights,dense_bias)
        mqtt_client.publish(self.topic,payload_bias )
        mqtt_client.publish(self.topic, payload_weights)
        print(f"gRPC publish: {payload_bias}")
        mqtt_client.disconnect()


#mqtt_pub_metadata()

