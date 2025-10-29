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


import datetime
import json
import threading

from utils import DataLoader

from utils import DataSaver
from utils import LeamPipeline
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
import base64
import logging

class FederatedLearningServicer(model_pb2_grpc.FederatedLearningServicer):
    def __init__(self, data_dir=None, mqtt_client=None):
        self.data_dir = data_dir
        self.model_parameters_list = np.empty((0, 64), dtype=np.float32)
        self.model_labels_list = np.empty((0,), dtype=np.int32)
        self.client_id = None
        self._lock = threading.Lock()
        # optional: keep a reference to mqtt client so pb_to_mqtt can publish
        self.mqtt_client = mqtt_client

    def pb_to_mqtt(self, model_parms1, model_parms2, client_id, publish=False, topic=FEDER_PUBLISH):
        """
        Convert numpy arrays to protobuf and optionally publish via self.mqtt_client.
        Returns tuple(payload_weights_bytes, payload_bias_bytes)
        """
        # Normalize to numpy arrays
        par1 = np.asarray(model_parms1)
        par2 = np.asarray(model_parms2)

        # build weight message
        msg_w = model_pb2.ModelParams()
        msg_w.param_type = model_pb2.CLASSIFIER_WEIGHT
        msg_w.values.extend(par1.flatten().astype(float).tolist())
        msg_w.client_id = client_id or ""
        payload_w = msg_w.SerializeToString()

        # build bias message
        msg_b = model_pb2.ModelParams()
        msg_b.param_type = model_pb2.CLASSIFIER_BIAS
        msg_b.values.extend(par2.flatten().astype(float).tolist())
        msg_b.client_id = client_id or ""
        payload_b = msg_b.SerializeToString()

        if publish and self.mqtt_client is not None:
            # publish as binary (or base64 if your broker/clients expect text)
            try:
                self.mqtt_client.publish(topic, payload_w)
                self.mqtt_client.publish(topic, payload_b)
                logging.info("Published model parameters to MQTT (bytes lengths: %d, %d)", len(payload_w), len(payload_b))
            except Exception as e:
                logging.exception("Failed to publish MQTT messages: %s", e)

        return payload_w, payload_b

    def GetUpdateStatus(self, request, context):
        return model_pb2.ServerResponse(
            message="Model update status fetched successfully.",
            update_successful=True,
            update_timestamp=int(time.time())
        )

    def federated_avg_from_DataLoder(self, data_dir, device_id):
        data_loader = DataLoader(data_dir=data_dir, device_id=device_id)
        pipeline = LeamPipeline(data_loader=data_loader)
        devices = pipeline.get_available_devices()
        federated_data = pipeline.get_federated_dataset(devices=devices, samples_per_device=500)
        return federated_data

    def federated_avg(self, data_dir, device_id):
        features, labels = self.federated_avg_from_DataLoder(data_dir, device_id)
        return self.pb_to_mqtt(features, labels, device_id)

    def UploadModelParams(self, request, context):
        client_id = getattr(request, "client_id", "")
        logging.info("收到来自客户端 %s 的参数", client_id)
        try:
            client_params = list(request.values)  # request.values is iterable
            if not client_params:
                raise ValueError("Empty values received in UploadModelParams")

            GROUP_SIZE = 65
            if len(client_params) % GROUP_SIZE != 0:
                raise ValueError(f"values length {len(client_params)} not divisible by GROUP_SIZE={GROUP_SIZE}")

            num_groups = len(client_params) // GROUP_SIZE
            data = np.array(client_params, dtype=np.float32).reshape(num_groups, GROUP_SIZE)
            labels_array = data[:, 0].astype(np.int32)
            params_array = data[:, 1:65].astype(np.float32)

            # thread-safe append
            with self._lock:
                if self.model_parameters_list.size == 0:
                    # ensure shape (0, 64)
                    self.model_parameters_list = np.empty((0, params_array.shape[1]), dtype=np.float32)

                if params_array.shape[1] != self.model_parameters_list.shape[1]:
                    logging.warning("维度不匹配，重置存储列表：%s -> %s", self.model_parameters_list.shape, params_array.shape)
                    self.model_parameters_list = np.empty((0, params_array.shape[1]), dtype=np.float32)
                    self.model_labels_list = np.empty((0,), dtype=np.int32)

                self.model_parameters_list = np.vstack([self.model_parameters_list, params_array])
                self.model_labels_list = np.concatenate([self.model_labels_list, labels_array])

                n_saved = self.model_parameters_list.shape[0]
                logging.info("Received model_parameters_list rows: %d", n_saved)

                if n_saved >= 10:
                    data_gen = DataSaver(self.data_dir, client_id)
                    data_gen.save_features(features=self.model_parameters_list, labels=self.model_labels_list)
                    # reset
                    self.model_parameters_list = np.empty((0, params_array.shape[1]), dtype=np.float32)
                    self.model_labels_list = np.empty((0,), dtype=np.int32)
                    logging.info("Model parameters saved and reset buffers")

            return model_pb2.ServerResponse(
                message="Model parameters successfully updated.",
                update_successful=True,
                update_timestamp=int(time.time())
            )

        except Exception as e:
            logging.exception("Error during UploadModelParams")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return model_pb2.ServerResponse(
                message=f"Model parameters not updated. client:{client_id} error:{str(e)}",
                update_successful=False,
                update_timestamp=int(time.time())
            )


class MqttClientServer(mqtt_client.Client):
    def __init__(self,fserv=None ,mqtt_broker=None,mqtt_port=None,data_dir=None):
        super().__init__()
        self.data_dir = data_dir
        self.client_id =None
        self.mqtt_broker = mqtt_broker
        self.mqtt_port = mqtt_port
        self.fserv = fserv
        #self.on_connect = self.on_connect
        #self.on_message = self.on_message

    def start_connect(self):
        username = "tim"  # 替换为你的 MQTT 用户名
        password = "tim"  # 替换为你的 MQTT 密码
        self.username_pw_set(username, password)  # 设置用户名和密码
        reconnect_timeout_ms = 10000  # 10秒的重连超时
        self.reconnect_delay_set(min_delay=1, max_delay=10)  # 设置重连延迟（最小1秒，最大10秒）


        result = self.connect(self.mqtt_broker, self.mqtt_port, 60)
        if result != 0:
            print(f"[MQTT] connect() returned {result} — check broker/port/auth")
        else:
            print(f" [MQTT] Sucessfully connected to {self.mqtt_broker}:{self.mqtt_port}")
        self.loop_start()
        #print(f"[MQTT] connected to {self.mqtt_broker} {self.mqtt_port}")


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

        # Load Fisher matrix ewc_assets.npz fisher_matrix.npz
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
            #print(f"[MQTT] 发布分片 {i + 1}/{total_chunks}, result={result}")

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
    #@classmethod
    def on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            print("Connected to MQTT broker successfully!")
            # 连接成功后，订阅一个主题
            result, mid = client.subscribe(GRPC_SUBSCRIBE)
            print(f" Subscribe result: {result}, mid: {mid}")

        else:
            print("Failed to connect, return code:", rc)
    #@classmethod
    def on_message(self,client,  userdata, msg):
        global client_request_code
        print(f"[MQTT] Received on {msg.topic}: {msg.payload.decode()}")
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

