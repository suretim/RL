import paho.mqtt.client as mqtt_client
import numpy as np
import tensorflow as tf
import os
import time
import json
import grpc
import model_pb2
import model_pb2_grpc

import math
#MQTT_BROKER = "127.0.0.1"

GRPC_SERVER = "127.0.0.1:50051"
MQTT_PORT = 1883
FEDER_PUBLISH = "federated_model/parameters"
WEIGHT_FISH_PUBLISH = "ewc/weight_fisher"
FISH_SHAP_PUBLISH = "ewc/layer_shapes"

GRPC_SUBSCRIBE = "grpc_sub/weights"
EWC_ASSETS="../lstm/ewc_assets"
DATA_DIR = "../../../../data"
client_request_code=1
class MqttClientServer(mqtt_client.Client):
    def __init__(self,fserv=None ,mqtt_broker=None,mqtt_port=None,data_dir=None):
        super().__init__()
        self.data_dir = data_dir
        self.client_id =None
        self.mqtt_broker = mqtt_broker
        self.mqtt_port = mqtt_port
        self.fserv = fserv
        #self.on_connect = self._on_connect
        #self.on_message = self._on_message


    def start_connect(self):
        self.connect(self.mqtt_broker, self.mqtt_port, 60)

        self.loop_start()
        print(f"[MQTT] connected to {self.mqtt_broker}{self.mqtt_port}")
    def save_fisher_matrix_to_bin(self ,fisher_matrix, bin_file_path):
        # Open the binary file in write mode
        with open(bin_file_path, 'wb') as bin_file:
            for matrix in fisher_matrix:
                # Convert each matrix (numpy array) to raw bytes
                matrix_bytes = matrix.numpy().tobytes()
                bin_file.write(matrix_bytes)  # Write the bytes to the file
        print(f"Fisher matrix saved to {bin_file_path}")

    def load_ewc_assets(self,save_dir=EWC_ASSETS):
        fisher_data = np.load(f"{save_dir}/fisher_matrix.npz")
        fisher_matrix = [tf.constant(arr) for arr in fisher_data.values()]
        return fisher_matrix

    def pubish_fisher_matrix(self ,client, topic, bin_file_path):
        with open(bin_file_path, 'rb') as f:
            payload = f.read()  # Read the binary content of the .bin file
            client.publish(topic, payload)  # Send the binary data as the MQTT message
            print(f"Fisher matrix sent to topic {topic}")

    def save_ewc_assets_to_bin(self ,save_dir=EWC_ASSETS):
        # Load model weights
        # model.load_weights(os.path.join(save_dir, "model_weights.h5"))

        # Load Fisher matrix
        fisher_data = np.load(os.path.join(save_dir, "fisher_matrix.npz"))
        fisher_matrix = [tf.constant(arr) for arr in fisher_data.values()]

        print(f"EWC assets loaded from {save_dir}")
        self.save_fisher_matrix_to_bin(fisher_matrix ,os.path.join(save_dir, "fisher_matrix.bin"))

        return fisher_matrix



    def publish_message(self):
        fisher_matrix = self.load_ewc_assets(save_dir=EWC_ASSETS)

        # 转成 bytes
        message = b''.join([arr.numpy().tobytes() for arr in fisher_matrix])

        # 分片大小（4KB）
        chunk_size = 480
        total_chunks = math.ceil(len(message) / chunk_size)

        print(f"[MQTT] Fisher matrix 大小={len(message)} bytes, 分成 {total_chunks} 片")

        for i in range(total_chunks):
            chunk = message[i * chunk_size:(i + 1) * chunk_size]

            # 包一层 JSON，带上分片信息
            payload = {
                "seq_id": i,
                "total": total_chunks,
                "data": chunk.hex()  # 转 hex 避免二进制 publish 出现乱码
            }

            # 发布分片
            result = self.publish(WEIGHT_FISH_PUBLISH, json.dumps(payload), qos=1)
            print(f"[MQTT] 发布分片 {i + 1}/{total_chunks}, result={result}")

    def publish_messagex(self):
        global client_request_code
        while True:
        #if client_request_code >= 2:
            fisher_matrix = self.load_ewc_assets(save_dir=EWC_ASSETS)
            message = b''.join([arr.numpy().tobytes() for arr in fisher_matrix])
            # message=load_ewc_assets(model, save_dir="../lstm/ewc_assets")
            # message = f"定时消息 @ {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            # 发布消息

            result = self.publish(WEIGHT_FISH_PUBLISH, message, qos=1)
            # 检查发布状态
            if result.rc == mqtt_client.MQTT_ERR_SUCCESS:
                # print(f"已发布: {message} → [{WEIGHT_FISH_PUBLISH}]")
                print(f"已发布:   [{WEIGHT_FISH_PUBLISH}]", client_request_code)
                client_request_code = 0
            else:
                print(f"发布失败，错误码: {result.rc}")
        # 等待180秒
        time.sleep(30)
        client_request_code = client_request_code + 1

    # MQTT 客户端回调函数
    @classmethod
    def _on_connect(cls, client, userdata, flags, rc):
        if rc == 0:
            print("Connected to MQTT broker successfully!")
            # 连接成功后，订阅一个主题
            result, mid = client.subscribe(GRPC_SUBSCRIBE)
            print(f" Subscribe result: {result}, mid: {mid}")

        else:
            print("Failed to connect, return code:", rc)
    @classmethod
    def _on_message(cls,client,  userdata, msg):
        global client_request_code
        #print(f"[MQTT] Received on {msg.topic}: {msg.payload.decode()}")
        try:
            # 尝试解析 JSON 并提取参数
            message = json.loads(msg.payload.decode())
            #client_request = message.get('client_request', '0')
            #client_id = message.get('client_id', '1')


            # 提取字段
            client_request = int(message.get("client_request", 0))
            client_id = int(message.get("client_id", 0))

            print(f"[MQTT] client_request={client_request}, client_id={client_id}")

            if client_request == 1:
                client.publish_message()
                print("[MQTT] client_request=1,publish ACK")
                return

            fea_weights = message.get("fea_weights", [])
            fea_labels = message.get("fea_labels", [])


            if not isinstance(fea_weights, list):
                fea_weights = [fea_weights]
            if not isinstance(fea_labels, list):
                fea_labels = [fea_labels]
            # 拼接 + flatten
            fea_vec = np.array(fea_labels + fea_weights, dtype=float).flatten().tolist()

            print(f"[TEST] fea_vec 长度 = {len(fea_vec)}")
            print(f"[TEST] 前 5 个值 = {fea_vec[:5]}")
            # load_ewc_assets(model, save_dir=EWC_ASSETS)
            # pubish_fisher_matrix(client=client, topic=MSG_PUBLISH, bin_file_path=os.path.join(EWC_ASSETS, "fisher_matrix.bin"))

            # 建立 gRPC 通信
            grpc_channel = grpc.insecure_channel(GRPC_SERVER)
            stub = model_pb2_grpc.FederatedLearningStub(grpc_channel)

            # 构建 gRPC 请求
            request = model_pb2.ModelParams(client_id=client_id, values=fea_vec)

            # 调用远程接口
            response = stub.UploadModelParams(request)
            if response.update_successful is True:
                payload_weights, payload_bias = client.fserv.federated_avg(data_dir=DATA_DIR, device_id=client_id)
                client.publish(FEDER_PUBLISH, payload_weights)
                client.publish(FEDER_PUBLISH, payload_bias)
            print(f"gRPC server response: {response.message}")

        except json.JSONDecodeError as e:
            print(f"Failed to decode MQTT message as JSON: {e}")
        except grpc.RpcError as e:
            print(f"gRPC communication failed: {e.details()} (code: {e.code()})")
        except Exception as e:
            print(f"Unexpected error in on_message: {e}")

