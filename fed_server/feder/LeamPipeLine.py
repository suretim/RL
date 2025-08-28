import h5py
import tensorflow as tf

import numpy as np
from pathlib import Path
import json
from datetime import datetime
from DataLoader import DataLoader as loader
#import federated_cifar

class LeamPipeline:
    def __init__(self,data_dir=None, data_loader =None):
        """

        :type data_loader: DataLoader
        """
        self.data_dir = data_dir
        self.data_loader = data_loader
        self.cache = {}  # 可選的緩存機制
        if(data_loader is None):
            self.data_loader=loader(data_dir="data", device_id="esp32_001")
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

        features, labels = loader.load_data()


        self.cache[device_id] = (features, labels)
        return features, labels

    def get_federated_dataset(self, devices=None, samples_per_device=None):
        """創建聯邦學習數據集"""
        federated_data = []

        for device_id in devices:
            features, labels = loader.load_data(
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

            federated_data.append((features, labels))

        return federated_data

    def get_centralized_dataset(self, devices=None, test_size=0.2):
        """創建集中式訓練數據集"""
        if devices is None:
            devices = self.get_available_devices()

        all_features = []
        all_labels = []

        for device_id in devices:
            features, labels = loader.load_data(
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
                batch_features, batch_labels =  loader.parallel_load(
                                                    self.data_loader,
                                                    files=batch_files,
                                                    max_workers=batch_size
                                                )

                # 在這裡添加您的處理代碼
                # 例如: 訓練模型、計算統計量等
                mean_features = np.mean(batch_features, axis=0)
                print(f"本批次特徵均值: {mean_features[:5]}...")  # 顯示前5維


if __name__ == '__main__':
    '''
    model = tf.keras.models.load_model("data\client_002_20250808_184020.h5")
    model.summary()
    with h5py.File("data\client_002_20250808_184020.h5", "r") as f:
        features = f["features"][:]

    print(features)
    print("Shape:", features.shape)
    print("Dtype:", features.dtype)
    '''
    # 使用示例
    data_loader = loader(data_dir="..\..\..\..\data", device_id="client_001")
    pipeline = LeamPipeline(data_loader=data_loader)
    devices = pipeline.get_available_devices()
    federated_data = pipeline.get_federated_dataset(devices=devices, samples_per_device=500)
    target_device="client_002"
    pipeline.load_available_devices(target_device)
    pipeline.load_available_devices()

    #from concurrent.futures import ThreadPoolExecutor

    #from my_load import MY_LOAD as loader

    # 创建数据加载器
    #data_loader = loader(data_dir="data", device_id="esp32_001")

    #from pathlib import Path

    # 獲取數據目錄下所有 .h5 文件





