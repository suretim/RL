# federated_server.py
"""
整合的 gRPC + MQTT 服务端示例（已改进并可直接运行）。
- 启动 gRPC server 提供 UploadModelParams RPC
- 启动 MQTT client，订阅 GRPC_SUBSCRIBE 主题并在收到消息后调用 gRPC
注意: 请根据实际环境调整常量（MQTT_BROKER / MQTT_PORT / GRPC_SERVER / DATA_DIR / EWC_ASSETS）
"""

import os
import time
import json
import math
import logging
import threading
import base64
import numpy as np

# 如果不需要 TensorFlow 的部分，请注释下面这行
import tensorflow as tf

import grpc
import paho.mqtt.client as mqtt_client

import model_pb2
import model_pb2_grpc

# 本地工具模块（需在 PYTHONPATH 下）
from utils import DataLoader, DataSaver, LeamPipeline

# ----- 配置 -----
MQTT_BROKER = "192.168.30.86"
MQTT_PORT = 1883
GRPC_SERVER = "127.0.0.1:50051"

FEDER_PUBLISH = "federated_model/parameters"
WEIGHT_FISH_PUBLISH = "ewc/weight_fisher"
FISH_SHAP_PUBLISH = "ewc/layer_shapes"
GRPC_SUBSCRIBE = "grpc_sub/weights"

EWC_ASSETS = "../lstm/ewc_assets"
DATA_DIR = "../../../data"

# 每组长度：label(1) + features(64) = 65
GROUP_SIZE = 65

# 保存阈值：当累计行数 >= FLUSH_ROWS 时写入 DataSaver 并清空 buffers
FLUSH_ROWS = 10

# ----- 日志 -----
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


# ---------------------------
# FederatedLearningServicer
# ---------------------------
class FederatedLearningServicer(model_pb2_grpc.FederatedLearningServicer):
    def __init__(self, data_dir=None, mqtt_client=None):
        self.data_dir = data_dir or DATA_DIR
        # buffers for accumulating features / labels
        self.model_parameters_list = np.empty((0, 64), dtype=np.float32)
        self.model_labels_list = np.empty((0,), dtype=np.int32)
        self._lock = threading.Lock()
        self.mqtt_client = mqtt_client  # optional reference to mqtt client for publishing via pb_to_mqtt

    def pb_to_mqtt(self, model_parms1, model_parms2, client_id="", publish=False, topic=FEDER_PUBLISH):
        """
        Convert numpy arrays to model_pb2.ModelParams and optionally publish using self.mqtt_client.
        Returns the serialized payloads (weights_bytes, bias_bytes).
        """
        par1 = np.asarray(model_parms1)
        par2 = np.asarray(model_parms2)

        msg_w = model_pb2.ModelParams()
        msg_w.param_type = model_pb2.CLASSIFIER_WEIGHT
        msg_w.values.extend(par1.flatten().astype(float).tolist())
        msg_w.client_id = str(client_id)
        payload_w = msg_w.SerializeToString()

        msg_b = model_pb2.ModelParams()
        msg_b.param_type = model_pb2.CLASSIFIER_BIAS
        msg_b.values.extend(par2.flatten().astype(float).tolist())
        msg_b.client_id = str(client_id)
        payload_b = msg_b.SerializeToString()

        if publish and (self.mqtt_client is not None):
            try:
                # publish binary payloads directly
                self.mqtt_client.publish(topic, payload_w, qos=1)
                self.mqtt_client.publish(topic, payload_b, qos=1)
                logging.info("Published model params to MQTT (%d, %d bytes)", len(payload_w), len(payload_b))
            except Exception:
                logging.exception("Failed to publish pb_to_mqtt payloads")

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
        """
        RPC 接口：客户端上传模型特征（values）。
        假设 client 每次发送的 values 是若干组 (label + 64-d features) 串联而成。
        """
        client_id = getattr(request, "client_id", "")
        logging.info("Received UploadModelParams from client_id=%s (values_len=%d)", client_id, len(request.values))

        try:
            client_params = list(request.values)
            if not client_params:
                raise ValueError("Empty values received.")

            if len(client_params) % GROUP_SIZE != 0:
                raise ValueError(f"values length {len(client_params)} not divisible by GROUP_SIZE={GROUP_SIZE}")

            num_groups = len(client_params) // GROUP_SIZE
            data = np.array(client_params, dtype=np.float32).reshape(num_groups, GROUP_SIZE)
            labels_array = data[:, 0].astype(np.int32)
            params_array = data[:, 1:1 + 64].astype(np.float32)  # 64 features

            with self._lock:
                # if first time, ensure width matches
                if self.model_parameters_list.size == 0:
                    self.model_parameters_list = np.empty((0, params_array.shape[1]), dtype=np.float32)
                    self.model_labels_list = np.empty((0,), dtype=np.int32)

                if params_array.shape[1] != self.model_parameters_list.shape[1]:
                    logging.warning("Feature dimension mismatch: reset buffers %s -> %s",
                                    self.model_parameters_list.shape, params_array.shape)
                    self.model_parameters_list = np.empty((0, params_array.shape[1]), dtype=np.float32)
                    self.model_labels_list = np.empty((0,), dtype=np.int32)

                # append
                self.model_parameters_list = np.vstack([self.model_parameters_list, params_array])
                self.model_labels_list = np.concatenate([self.model_labels_list, labels_array])

                n_saved = self.model_parameters_list.shape[0]
                logging.info("Buffered rows: %d", n_saved)

                # flush to disk when enough rows collected
                if n_saved >= FLUSH_ROWS:
                    data_gen = DataSaver(self.data_dir, client_id)
                    data_gen.save_features(features=self.model_parameters_list, labels=self.model_labels_list)
                    logging.info("Saved %d rows via DataSaver to %s (client %s)", n_saved, self.data_dir, client_id)
                    # reset
                    self.model_parameters_list = np.empty((0, params_array.shape[1]), dtype=np.float32)
                    self.model_labels_list = np.empty((0,), dtype=np.int32)

            return model_pb2.ServerResponse(
                message="Model parameters successfully processed.",
                update_successful=True,
                update_timestamp=int(time.time())
            )

        except Exception as e:
            logging.exception("UploadModelParams error")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return model_pb2.ServerResponse(
                message=f"Model parameters not updated. client:{client_id} error:{str(e)}",
                update_successful=False,
                update_timestamp=int(time.time())
            )


# ---------------------------
# MqttClientServer
# ---------------------------
class MqttClientServer(mqtt_client.Client):
    def __init__(self, client_id=None, fserv: FederatedLearningServicer = None,
                 mqtt_broker=MQTT_BROKER, mqtt_port=MQTT_PORT, ewc_assets=EWC_ASSETS):
        super().__init__(client_id=client_id or "")
        self.mqtt_broker = mqtt_broker
        self.mqtt_port = mqtt_port
        self.fserv = fserv
        self.ewc_assets = ewc_assets
        # internal state
        self._stop_flag = threading.Event()

        # Bind instance methods for callbacks
        self.on_connect = self._on_connect
        self.on_message = self._on_message

    def start_connect(self):
        logging.info("Connecting to MQTT broker %s:%d ...", self.mqtt_broker, self.mqtt_port)
        self.connect(self.mqtt_broker, self.mqtt_port, keepalive=60)
        self.loop_start()
        logging.info("MQTT loop started")

    def stop_connect(self):
        self._stop_flag.set()
        try:
            self.disconnect()
            self.loop_stop()
            logging.info("MQTT stopped")
        except Exception:
            logging.exception("Error stopping MQTT")

    # --- EWC helpers (I/O helpers for fisher matrix) ---
    def load_ewc_assets(self, save_dir=None):
        save_dir = save_dir or self.ewc_assets
        path = os.path.join(save_dir, "fisher_matrix.npz")
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        fisher_data = np.load(path)
        fisher_matrix = [tf.constant(arr) for arr in fisher_data.values()]
        logging.info("Loaded EWC assets from %s", path)
        return fisher_matrix

    def save_fisher_matrix_to_bin(self, fisher_matrix, bin_file_path):
        with open(bin_file_path, "wb") as fout:
            for arr in fisher_matrix:
                # arr is a tf.Tensor -> convert to numpy
                nd = arr.numpy() if hasattr(arr, "numpy") else np.asarray(arr)
                fout.write(nd.tobytes())
        logging.info("Saved fisher matrix bin to %s", bin_file_path)

    def publish_fisher_matrix_chunks(self, topic=WEIGHT_FISH_PUBLISH, chunk_size=480):
        """
        Example: split binary into hex-encoded chunks and publish JSON messages.
        Use this if raw binary over MQTT is problematic for clients/brokers.
        """
        fisher_matrix = self.load_ewc_assets()
        message = b"".join([arr.numpy().tobytes() for arr in fisher_matrix])
        total = len(message)
        total_chunks = math.ceil(total / chunk_size)
        logging.info("Publishing fisher matrix total %d bytes in %d chunks", total, total_chunks)

        for i in range(total_chunks):
            chunk = message[i * chunk_size:(i + 1) * chunk_size]
            payload = {"seq_id": i, "total": total_chunks, "data_hex": chunk.hex()}
            # publish JSON string
            self.publish(topic, json.dumps(payload), qos=1)

    # --- MQTT callbacks ---
    def _on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            logging.info("Connected to MQTT broker OK (rc=%d)", rc)
            # subscribe to control topic for GRPC requests
            client.subscribe(GRPC_SUBSCRIBE)
            logging.info("Subscribed to %s", GRPC_SUBSCRIBE)
        else:
            logging.error("Failed to connect to MQTT broker (rc=%d)", rc)

    def _on_message(self, client, userdata, msg):
        """
        Expect JSON payload with keys:
          - client_request: int (1==request fisher publish)
          - client_id: int or str
          - fea_weights: list or scalar
          - fea_labels: list or scalar
        """
        try:
            payload = msg.payload
            # try parse as text JSON first
            try:
                data = json.loads(payload.decode("utf-8"))
            except Exception:
                # if not JSON, warn and return
                logging.warning("Received non-JSON payload on %s (%d bytes)", msg.topic, len(payload))
                return

            client_request = int(data.get("client_request", 0))
            client_id = data.get("client_id", "")
            logging.info("MQTT message client_request=%s client_id=%s", client_request, client_id)

            if client_request == 1:
                # publish fisher matrix chunks (example)
                threading.Thread(target=self.publish_fisher_matrix_chunks, daemon=True).start()
                logging.info("Spawned fisher matrix publish thread")
                return

            # parse features & labels from JSON
            fea_weights = data.get("fea_weights", [])
            fea_labels = data.get("fea_labels", [])

            # ensure lists
            if not isinstance(fea_weights, list):
                fea_weights = [fea_weights]
            if not isinstance(fea_labels, list):
                fea_labels = [fea_labels]

            # create concatenated vector: label(s) followed by weights
            fea_vec = list(map(float, fea_labels)) + list(map(float, fea_weights))
            logging.debug("fea_vec len=%d first5=%s", len(fea_vec), fea_vec[:5])

            # call gRPC UploadModelParams
            with grpc.insecure_channel(GRPC_SERVER) as channel:
                stub = model_pb2_grpc.FederatedLearningStub(channel)
                request = model_pb2.ModelParams(client_id=str(client_id), values=fea_vec)
                response = stub.UploadModelParams(request)
                logging.info("gRPC UploadModelParams response: %s success=%s", response.message, response.update_successful)

                # if gRPC processed and we want to publish aggregated model from server side:
                if response.update_successful and self.fserv is not None:
                    try:
                        payload_w, payload_b = self.fserv.federated_avg(data_dir=DATA_DIR, device_id=client_id)
                        # publish aggregated params to FEDER_PUBLISH
                        # NOTE: payload_w/payload_b are serialized protobuf bytes
                        self.publish(FEDER_PUBLISH, payload_w, qos=1)
                        self.publish(FEDER_PUBLISH, payload_b, qos=1)
                        logging.info("Published aggregated model to %s", FEDER_PUBLISH)
                    except Exception:
                        logging.exception("Failed to federated_avg & publish")
        except Exception as e:
            logging.exception("Error in _on_message: %s", e)


# ---------------------------
# gRPC server runner
# ---------------------------
def run_grpc_server(fserv, host="[::]:50051", max_workers=10):
    server = grpc.server(thread_pool=futures.ThreadPoolExecutor(max_workers=max_workers))
    model_pb2_grpc.add_FederatedLearningServicer_to_server(fserv, server)
    server.add_insecure_port(host)
    server.start()
    logging.info("gRPC server started on %s", host)
    return server


# ---------------------------
# main
# ---------------------------
from concurrent import futures


def main():
    # create servicer (without mqtt yet)
    fserv = FederatedLearningServicer(data_dir=DATA_DIR)

    # create mqtt client and inject fserv
    mqttc = MqttClientServer(client_id="federated_server", fserv=fserv, mqtt_broker=MQTT_BROKER, mqtt_port=MQTT_PORT)
    # allow fserv to publish via mqtt if desired
    fserv.mqtt_client = mqttc

    try:
        # start MQTT
        mqttc.start_connect()

        # start gRPC server (non-blocking)
        grpc_server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
        model_pb2_grpc.add_FederatedLearningServicer_to_server(fserv, grpc_server)
        grpc_server.add_insecure_port("[::]:50051")
        grpc_server.start()
        logging.info("gRPC started at [::]:50051")

        logging.info("Server up. Ctrl+C to stop.")
        # block
        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        logging.info("KeyboardInterrupt received. Shutting down...")
    except Exception:
        logging.exception("Unhandled exception in main loop")
    finally:
        # stop mqtt and grpc
        try:
            mqttc.stop_connect()
        except Exception:
            pass
        try:
            grpc_server.stop(5)
            logging.info("gRPC stopped")
        except Exception:
            pass
        logging.info("Exiting.")


if __name__ == "__main__":
    main()
