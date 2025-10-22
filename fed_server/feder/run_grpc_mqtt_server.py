import grpc
import model_pb2
import model_pb2_grpc
from concurrent import futures
import time
import datetime
import json
import threading
import numpy as np

from utils import DataLoader

from utils import DataSaver
from utils import LeamPipeline
from util_FedServer import MqttClientServer,FederatedLearningServicer
import os
import sys
import argparse

# MQTT配置

#MQTT_BROKER = "192.168.0.57"

#conda activate my_env
#cd C:\tim\aicam\main\fed_server\cloud_models
#python emqx_manager.py
#netstat -ano | findstr :18083
#fserv=None
MQTT_PORT = 1883
MQTT_BROKER = "192.168.30.86"
GRPC_SUBSCRIBE = "grpc_sub/weights"
FEDER_PUBLISH = "federated_model/parameters"
GRPC_SERVER = "127.0.0.1:50051"

EWC_ASSETS="../lstm/ewc_assets"
DATA_DIR = "../../../data"

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

class xFederatedLearningServicer(model_pb2_grpc.FederatedLearningServicer):
    def __init__(self,data_dir=None):
        self.data_dir = data_dir
        self.model_parameters_list = np.empty((0, 64))
        self.model_labels_list = np.empty((0,))
        self.client_id=None

    # 用于更新模型的函数
    @classmethod
    def pb_to_mqtt(cls,model_parms1,model_parms2,client_id):
        # 如果是 numpy 数组，先转成列表
        par1 =model_parms1
        par2 =model_parms2
        if isinstance(par1, np.ndarray):
            par1 = par1.tolist()
        elif isinstance(par1, list) and isinstance(par1[0], np.ndarray):
            par1 = [w.tolist() for w in par1]
        if isinstance(par2, np.ndarray):
            par2 = par2.tolist()
        elif isinstance(par2, list) and isinstance(par2[0], np.ndarray):
            par2 = [w.tolist() for w in par2]

        # 构建消息
        msg_weights = model_pb2.ModelParams()
        msg_weights.param_type = model_pb2.CLASSIFIER_WEIGHT
        msg_weights.values.extend(par1.flatten().tolist())
        msg_weights.client_id = client_id  # 可选设置 client_id
        payload_weights = msg_weights.SerializeToString()
        #mqtt_client.publish(FEDER_PUBLISH, payload_weights)
        print(f"Published model parameters to MQTT: {payload_weights}")
        msg_bias = model_pb2.ModelParams()
        msg_bias.param_type = model_pb2.CLASSIFIER_BIAS
        msg_bias.values.extend(par2.flatten().tolist())
        msg_bias.client_id = client_id  # 可选设置 client_id
        payload_bias = msg_bias.SerializeToString()
        #mqtt_client.publish(FEDER_PUBLISH, payload_bias)
        print(f"Published model parameters to MQTT: {payload_bias}")
        # 打包为 JSON 格式
        # weights_data = {
        #    "mqtrx_weights": model_parameters,
        # "metadata": {
        #     "num_classes": 5,
        #     "input_shape": 64
        # }
        # }
        """通过 MQTT 发布全局模型参数"""
        # payload = json.dumps(weights_data)  # 序列化为字符串
        return payload_weights,payload_bias

    def GetUpdateStatus(self, request, context):
        # 假设总是成功并返回状态
        return model_pb2.ServerResponse(
            message="Model update status fetched successfully.",
            update_successful=True,
            update_timestamp=int(time.time())
        )

    # 假设你希望每次收到一个 client 模型参数都加入一个列表后聚合


    def federated_avg_from_DataLoder(self,data_dir,device_id):
        """
        简单的 FedAvg 实现：对多个客户端上传的模型参数（float 数组）取平均
        参数:
            model_parameters_list: List of List[float]
        返回:
            List[float]: 平均后的模型参数
        """
        data_loader = DataLoader(data_dir=data_dir, device_id=device_id)
        pipeline = LeamPipeline(data_loader =data_loader)
        devices = pipeline.get_available_devices()
        federated_data = pipeline.get_federated_dataset(devices=devices, samples_per_device=500)

        #pipeline.load_available_devices(device_id)
        #pipeline.load_available_devices()
        return federated_data

    def federated_avg(self,data_dir,device_id):
        '''
        if not self.model_parameters_list:
            #raise ValueError("model_parameters_list is empty")
            avg_params,bias=self.federated_avg_from_DataLoder(data_dir,device_id)
        else:
            num_clients = len(self.model_parameters_list)
            num_params = len(self.model_parameters_list[0])

            # 初始化为 0
            avg_params = [0.0] * num_params

            for params in self.model_parameters_list:
                for i in range(num_params):
                    avg_params[i] += params[i]

            # 求平均
            avg_params= [x / num_clients for x in avg_params]
            bias=self.model_labels_list
            # 发布新模型参数
        '''
        features, labels  = self.federated_avg_from_DataLoder(data_dir, device_id)
        #federated_data.append((features, labels))
        return self.pb_to_mqtt(features, labels,device_id)



    def UploadModelParams(self, request, context):
        """
        更新全局模型并通过 MQTT 发布
        """
        client_id=request.client_id
        print(f"收到来自客户端 {request.client_id} 的参数")
        try:
            client_params = list(list(request.values) ) # 需要转换为 list
            #print("Received model parameters: ", client_params)
            #print("request.client_id  :",client_id)
            #print("client_params 結構:", client_params)
            #print("第一行類型:", type(client_params[0]))
            GROUP_SIZE = 65
            num_groups = len(client_params) // GROUP_SIZE

            # 转换为 NumPy 数组并重新组织
            data = np.array(client_params, dtype=np.float32).reshape(num_groups, GROUP_SIZE)
            labels_array = data[:, 0].astype(np.int32)  # 所有行的第 0 列（标签）
            params_array = data[:, 1:65]  # 所有行的第 1 列之后（特征）
            # 使用示例
            #params_array =np.random.rand(100, 64).astype(np.float32)  # 模擬ESP32輸出 client_params[1:64]  #
            #labels_array = np.random.randint(0, 3, size=100) # 模擬ESP32輸出 client_params[0]  #
            #params_array = np.array(client_params[1:], dtype=np.float32)  # Convert to NumPy array
            #labels_array = np.array([ client_params[0]], dtype=np.float32)  # Convert to NumPy array

            #labels_array = np.array([x[0] for x in client_params], dtype=np.int32)
            # 提取所有行的第 1 列之后（特征）
            #params_array = np.array([x[1:] for x in client_params], dtype=np.float32)
            #print("Received labels_array: ", labels_array )
            #print("Received params_array: ", params_array )


            # 初始化存储列表（如果是第一次运行）
            if not hasattr(self, 'model_parameters_list'):
                self.model_parameters_list = np.empty((0, 64))  # 特征维度 64
                self.model_labels_list = np.empty((0,))  # 标签

            # 检查维度一致性
            if params_array.shape[1] != self.model_parameters_list.shape[1]:
                print(f"维度不匹配！重置存储列表。",params_array.shape[1] )
                self.model_parameters_list = np.empty((0, 64))
                self.model_labels_list = np.empty((0,))

            # 追加数据
            self.model_parameters_list = np.vstack((self.model_parameters_list, params_array))
            self.model_labels_list = np.concatenate((self.model_labels_list, labels_array))
            # 聚合
            #self.model_parameters_list.append(params_array)
            #self.model_labels_list.append(labels_array)
            print("Received model_parameters_list: ", self.model_parameters_list.shape[0])
            print("Received model_labels_list: ", self.model_labels_list.shape[0])

            if self.model_parameters_list.shape[0]>=10:
                #arravg = np.array(parameters_avg)
                #print("federated_avg parameters: ", arravg)
                #features = np.round(features, decimals=3)  # Round to 1 decimal
                #print("federated features: ", features)

                #data_dir = "../../../../data"
                #device_id = "client_003"
                data_gen = DataSaver(self.data_dir,client_id)

                data_gen.save_features(
                    features=self.model_parameters_list,
                    labels=self.model_labels_list
                    #metadata={}
                )
                self.model_parameters_list = np.empty((0, 64))
                self.model_labels_list = np.empty((0,))
                print("Model parameters successfully updated." )

            # 返回响应
            # return model_pb2.UpdateResponse(status="Success")
            success = True  # Let's assume the update is successful for this example
            timestamp = int(time.time())  # Get current timestamp

            # Return response with a success message and timestamp
            return model_pb2.ServerResponse(
                message="Model parameters successfully updated.",
                update_successful=success,
                update_timestamp=timestamp)

        except Exception as e:
            print("Error during UploadModelParams:", e)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            success = False  # Let's assume the update is successful for this example
            timestamp = int(time.time())  # Get current timestamp
            return model_pb2.ServerResponse(
                message="Model parameters none successfully updated."+client_id,
                update_successful=success,
                update_timestamp=timestamp)



#end of FederatedLearningServicer


def serve(server,fserv):


    model_pb2_grpc.add_FederatedLearningServicer_to_server(fserv, server)
    server.add_insecure_port('[::]:50051')
    print("gRPC server started at port 50051")
    server.start()

    server.wait_for_termination()




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


        serve(server, fserv)
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
