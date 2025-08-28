import h5py
import numpy as np 
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor


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