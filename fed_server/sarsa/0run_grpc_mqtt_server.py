#import grpc
#import model_pb2
#import model_pb2_grpc
from concurrent import futures
import time
import datetime
import json
import threading
import numpy as np
import os
import sys
import argparse

from utils_MqttClient import MqttClientServer
from utils_feder import *


# MQTT配置

#MQTT_BROKER = "192.168.0.57"

#GRPC_SUBSCRIBE = "grpc_sub/weights"
#FEDER_PUBLISH = "federated_model/parameters"
#GRPC_SERVER = "127.0.0.1:50051"
MQTT_PORT = 1883
MQTT_BROKER = "127.0.0.1"
EWC_ASSETS="ewc_assets"
DATA_DIR = "../../../../data/sarsa_data"

#define MQTT_TOPIC_PUB "grpc_sub/weights"
#define MQTT_TOPIC_SUB "federated_model/parameters"
#define WEIGHT_FISH_SUB "ewc/weight_fisher"
#define FISH_SHAP_SUB  "ewc/layer_shapes"

#client_request_code= 1
#Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
#python -m venv .venv
#.\.venv\Scripts\activate  # Windows PowerShell
# 或 source .venv/bin/activate  # Linux/macOS
# python -m grpc_tools.protoc --proto_path=./ --python_out=./ --grpc_python_out=./ model.proto
# start mqtt server D:\mqttserver\emqx-5.0.26-windows-amd64\bin\emqx.cmd
#model_params = []
#model_parameters_list = []
#new_model_parameters=[]


def serve_fserv(server,fserv):


    model_pb2_grpc.add_FederatedLearningServicer_to_server(fserv, server)
    server.add_insecure_port('[::]:50051')
    print("gRPC server started at port 50051")
    server.start()

    server.wait_for_termination()



#conda activate my_env
#cd C:\tim\aicam\main\fed_server\cloud_models
#python emqx_manager.py
#netstat -ano | findstr :18083
#fserv=None


def main(args):
    global MQTT_BROKER,MQTT_PORT, EWC_ASSETS,DATA_DIR
    MQTT_BROKER = args.mqtt_broker
    MQTT_PORT=args.mqtt_port
    DATA_DIR=args.data_dir
    EWC_ASSETS = args.ewc_assets
    try:
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
        fserv = FederatedLearningServicer(data_dir=DATA_DIR)

        # 创建 MQTT 客户端
        mqtt_client = MqttClientServer(fserv=fserv ,
                                       mqtt_broker=args.mqtt_broker,
                                       mqtt_port=args.mqtt_port,
                                       data_dir=args.ewc_assets)
        mqtt_client.on_connect = MqttClientServer._on_connect
        mqtt_client.on_message = MqttClientServer._on_message
        # 设置用户名和密码
        username = "tim"  # 替换为你的 MQTT 用户名
        password = "tim"  # 替换为你的 MQTT 密码
        mqtt_client.username_pw_set(username, password)  # 设置用户名和密码
        # 设置重连超时时间，单位为毫秒
        reconnect_timeout_ms = 10000  # 10秒的重连超时
        mqtt_client.reconnect_delay_set(min_delay=1, max_delay=10)  # 设置重连延迟（最小1秒，最大10秒）

        #mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 60)
        mqtt_client.start_connect()

        #subcribe_thread = threading.Thread(target=mqtt_subscribe, args=(mqtt_client,))

        #subcribe_thread.start()

        # 创建定时发布线程
        #publish_thread = threading.Thread(target=publish_message)
        #publish_thread.daemon = True  # 设为守护线程
        #publish_thread.start()


        serve_fserv(server, fserv)
        # --- 阻塞主线程，等待 gRPC 和 MQTT 消息 ---

        while True:
            time.sleep(1)  # 主线程空转，后台线程处理 MQTT 和 gRPC


    except KeyboardInterrupt:
        print("\n程序终止")
        sys.exit(0)
    except Exception as e:
        print(f"发生错误: {str(e)}")

    finally:
        mqtt_client.disconnect()
        mqtt_client.loop_stop()
        print("MQTT closed  netstat -ano | findstr :50051")

#python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. model.proto

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=DATA_DIR)
    parser.add_argument("--mqtt_broker", type=str, default=MQTT_BROKER)
    parser.add_argument("--mqtt_port", type=int, default=MQTT_PORT)

    parser.add_argument("--ewc_assets", type=str, default=EWC_ASSETS)

    args = parser.parse_args()

    main(args)
