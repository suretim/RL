import h5py
import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import grpc
import model_pb2
import model_pb2_grpc
import time

import json
from datetime import datetime

class DataSaver:
    def __init__(self, data_dir=None, device_id=None):
        self.data_dir = data_dir
        self.device_id = device_id
        if (data_dir is None):
            self.data_dir = Path("data")
        else:
            self.data_dir = Path(data_dir)
        if (device_id is None):
            self.device_id ="esp32_001"
        else:
            self.device_id = device_id

        self.model_parameters_list = np.empty((0, 64))
        self.model_labels_list = np.empty((0,))

    def save_features(self,
                            features: np.ndarray,
                            labels: np.ndarray):
        """
        保存ESP32特徵數據到HDF5文件
        參數:
            device_id: 設備唯一標識符
            features: 編碼器輸出 (n_samples, 64)
            labels: 對應標籤 (n_samples,)
            output_dir: 存儲目錄
            metadata: 附加元數據
        """
        # 創建目錄
        Path(self.data_dir).mkdir(parents=True, exist_ok=True)

        # 生成文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.data_dir}/{self.device_id}_{timestamp}.h5"
        #filename = f"{self.data_dir}/{self.device_id}.h5"

        # 保存到HDF5
        with h5py.File(filename, 'w') as f:
            if features.ndim == 0:  # Scalar check
                f.create_dataset("features", data=features)
            else:
                f.create_dataset("features", data=features, compression="gzip")

            # Same for labels
            if labels.ndim == 0:
                f.create_dataset("labels", data=labels)
            else:
                f.create_dataset("labels", data=labels, compression="gzip")

            # Save metadata (no compression)
            #if metadata:
            #    for key, value in metadata.items():
            #        f.create_dataset(f"metadata/{key}", data=value)

            # 保存主要數據
            #f.create_dataset("features", data=features, compression="gzip" )
            #f.create_dataset("labels", data=labels, compression="gzip" )

            # 保存元數據
            #if metadata is None:
            metadata = {}
            metadata.update({
                "device_id": self.device_id,
                "timestamp": timestamp,
                "num_samples": len(features),
                "feature_dim": features.shape[1]
            })

            #print("更新前:", dict(f.attrs))  # 检查原有属性
            f.attrs.update(metadata)
            print("metadata attr:", dict(f.attrs))  # 确认更新结果
            f.close()


        print(f"數據已保存到 {filename}")
        return filename

    def _normalize_features(self,features: np.ndarray) -> np.ndarray:
        """標準化特徵數據 (每個維度0均值1方差)"""
        mean = np.mean(features, axis=0)
        std = np.std(features, axis=0)
        return (features - mean) / (std + 1e-7)  # 避免除零


    def save_with_versioning(self,device_id, features, labels, output_dir="data", max_versions=5):
        """帶版本控制的數據保存"""
        # 查找現有版本
        existing_files = sorted(Path(output_dir).glob(f"{device_id}_v*.h5"))

        # 確定新版本號
        version = 1
        if existing_files:
            last_version = int(existing_files[-1].stem.split("_v")[-1])
            version = last_version + 1

        # 刪除舊版本
        if len(existing_files) >= max_versions:
            for file in existing_files[:-(max_versions - 1)]:
                file.unlink()

        # 保存新文件
        filename = f"{output_dir}/{device_id}_v{version}.h5"
        self.save_features(device_id, features, labels, filename)


class DataLoader:
    def __init__(self, data_dir, device_id=None):
        self.data_dir = Path(data_dir)
        self.device_id = device_id



    def sumeray_files(self,data_dir):
        files = list(Path(data_dir).glob("*.h5"))
        # 檢查文件列表
        print(f"找到 {len(files)} 個數據文件")
        for file in files[:3]:  # 顯示前3個文件
            print(file.name)
    @staticmethod
    def load_data(self,data_dir,device_id):
        # 查找該設備的所有數據文件
        if device_id is None:
            device_id = self.device_id
        if data_dir is None:
            data_dir = self.data_dir

        device_files = list(Path(data_dir).glob(f"{device_id}_*.h5"))

        if not device_files:
            raise FileNotFoundError(f"找不到設備 {device_id} 的數據")

        # 並行加載數據
        features, labels = self.parallel_load(device_files)
        return features, labels

    def parallel_load(self, files, max_workers=4):

        """並行加載多個文件"""
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(
                lambda f: self._load_and_preprocess_data(f.parent, f.stem.split("_")[0]), files
            ))

        features = np.concatenate([r[0] for r in results])
        labels = np.concatenate([r[1] for r in results])
        return features, labels

    def _load_and_preprocess_data(self, data_dir: str,
                                        client_id: str = None,
                                        max_samples: int = None):
        """
        從目錄加載ESP32數據並預處理
        參數:
            data_dir: 數據目錄
            client_id: 指定設備ID (None則加載所有)
            max_samples: 每設備最大樣本數 (None則全部)
        返回:
            (features, labels) 元組
        """
        data_files = []

        # 查找匹配文件
        if client_id is None:
            data_files = list(Path(data_dir).glob("*.h5"))
        else:
            data_files = list(Path(data_dir).glob(f"{client_id}_*.h5"))

        if not data_files:
            raise FileNotFoundError(f"找不到 {client_id} 的數據文件")

        all_features = []
        all_labels = []

        for file in data_files:
            with h5py.File(file, 'r') as f:
                # 讀取數據
                features = f["features"][:]
                labels = f["labels"][:]

                # 限制樣本數
                if max_samples is not None and len(features) > max_samples:
                    indices = np.random.choice(len(features), max_samples, replace=False)
                    features = features[indices]
                    labels = labels[indices]

                all_features.append(features)
                all_labels.append(labels)

        # 合併數據
        features = np.concatenate(all_features)
        labels = np.concatenate(all_labels)

        # 數據預處理
        mean = np.mean(features, axis=0)
        std = np.std(features, axis=0)
        features = (features - mean) / (std + 1e-7)
        # features = normalize_features(features)
        labels = labels.astype(np.int32)

        # 打亂數據
        shuffle_idx = np.random.permutation(len(features))
        features = features[shuffle_idx]
        labels = labels[shuffle_idx]

        return features, labels



    def _find_data_files(self):
        """查找匹配的数据文件"""
        pattern = f"{self.device_id}_*.h5" if self.device_id else "*.h5"
        files = list(self.data_dir.glob(pattern))
        if not files:
            raise FileNotFoundError(f"No data files found for {self.device_id}")
        return files



class LeamPipeline:
    def __init__(self,data_dir=None, data_loader =None):
        """

        :type data_loader: DataLoader
        """
        self.data_dir = data_dir
        self.data_loader = data_loader
        self.cache = {}  # 可選的緩存機制
        if(data_loader is None):
            self.data_loader=DataLoader(data_dir="data", device_id="esp32_001")
        if (data_dir is None):
            self.data_dir = Path(self.data_loader.data_dir)
        else:
            self.data_dir=Path(data_dir)
    def _normalize_features(self,features: np.ndarray) -> np.ndarray:
        """標準化特徵數據 (每個維度0均值1方差)"""
        mean = np.mean(features, axis=0)
        std = np.std(features, axis=0)
        return (features - mean) / (std + 1e-7)  # 避免除零


    def load_and_cash_device_data(self, device_id):
        """加載單個設備數據並緩存"""
        if device_id in self.cache:
            return self.cache[device_id]

        features, labels = DataLoader.load_data()


        self.cache[device_id] = (features, labels)
        return features, labels

    def get_federated_dataset(self, devices=None, samples_per_device=None):
        """創建聯邦學習數據集"""
        #federated_data = []
        features=[]
        labels=[]
        for device_id in devices:
            features, labels = DataLoader.load_data(
                self.data_loader,
                data_dir=None,
                device_id=device_id
            )

            if samples_per_device and len(features) > samples_per_device:
                indices = np.random.choice(
                    len(features),
                    samples_per_device,
                    replace=False
                )
                features = features[indices]
                labels = labels[indices]

            #federated_data.append((features, labels))

        return features, labels

    def get_centralized_dataset(self, devices=None, test_size=0.2):
        """創建集中式訓練數據集"""
        if devices is None:
            devices = self.get_available_devices()

        all_features = []
        all_labels = []

        for device_id in devices:
            features, labels = DataLoader.load_data(
                data_dir=None,
                device_id=device_id
            )
            all_features.append(features)
            all_labels.append(labels)

        X = np.concatenate(all_features)
        y = np.concatenate(all_labels)

        # 分割訓練/測試集
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        return (X_train, y_train), (X_test, y_test)

    def get_available_devices(self):
        """獲取所有可用設備ID"""
        files = self.data_dir.glob("*.h5")
        #return list({f.stem.split("_")[0]+"_" +f.stem.split("_")[1]for f in files})

        return list({f.stem.split("_")[0]  for f in files})

    def load_available_devices(self, target_device=None):
        # 只加載特定ESP32設備的數據 (例如設備ID為esp32_001)
        #if target_device == None:
        #    target_device = self.get_available_devices()[0]
        #    print(f"找到設備 {target_device} ")
        if target_device == None:
            target_device = self.data_loader.device_id
            print(f"找到設備 data_loader {target_device} ")
        files = self.data_dir.glob("*.h5")
        device_files = [f for f in files if f.stem.startswith(target_device)]

        if not device_files:
            print(f"找不到設備 {target_device} 的數據文件")
        else:

            print(f"設備 {target_device} 的數據:")
            #print(f"- 樣本數: {len(device_features)}")
            print(f"- 最新文件: {device_files[-1].name}")

        # 當數據量很大時，可以分批加載處理
            batch_size = 3  # 每次處理5個文件
            total_files = len(device_files)

            for i in range(0, total_files, batch_size):
                batch_files = device_files[i:i + batch_size]
                print(f"\n處理文件 {i + 1}-{min(i + batch_size, total_files)}/{total_files}")

                # 並行加載當前批次
                batch_features, batch_labels =  DataLoader.parallel_load(
                                                    self.data_loader,
                                                    files=batch_files,
                                                    max_workers=batch_size
                                                )

                # 在這裡添加您的處理代碼
                # 例如: 訓練模型、計算統計量等
                mean_features = np.mean(batch_features, axis=0)
                print(f"本批次特徵均值: {mean_features[:5]}...")  # 顯示前5維
        return mean_features



class FederatedLearningServicer(model_pb2_grpc.FederatedLearningServicer):
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

                #data_dir = "../../../data"
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


