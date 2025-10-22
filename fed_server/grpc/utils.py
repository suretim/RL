import h5py
import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

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




