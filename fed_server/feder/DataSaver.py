import h5py
import numpy as np
from pathlib import Path
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

    def UploadModelParams( self,request_values, context=None):
        """
        更新全局模型并通过 MQTT 发布
        """
        try:
            client_params = list(request_values)  # 需要转换为 list
            #print("Received model parameters: ", client_params[0])
            # 使用示例
            #params_array =np.random.rand(100, 64).astype(np.float32)  # 模擬ESP32輸出 client_params[1:64]  #
            #labels_array = np.random.randint(0, 3, size=100) # 模擬ESP32輸出 client_params[0]  #
            #params_array = np.array(client_params[:][1:], dtype=np.float32)  # Convert to NumPy array
            #labels_array = np.array([ client_params[:][0]], dtype=np.int32)  # Convert to NumPy array
            # 提取所有行的第 0 列（标签）
            labels_array = np.array([x[0] for x in client_params], dtype=np.int32)
            # 提取所有行的第 1 列之后（特征）
            params_array = np.array([x[1:] for x in client_params], dtype=np.float32)
            print("Received labels_array: ", labels_array )
            if params_array.shape[1] != self.model_parameters_list.shape[1]:
                self.model_parameters_list = np.empty((0, 64))
                self.model_labels_list = np.empty((0,))
                #raise ValueError(
                #    f"Expected {self.model_parameters_list.shape[1]} features, got {params_array.shape[1]}")
            else:
                # 使用np.vstack进行垂直堆叠
                self.model_parameters_list = np.vstack((self.model_parameters_list, params_array))
                self.model_labels_list = np.concatenate((self.model_labels_list, labels_array))


            # 聚合
            #self.model_parameters_list.append(params_array)
            #self.model_labels_list.append(labels_array)

            if self.model_parameters_list.shape[0]>=2:
                #parameters_avg = self.federated_avg(self.model_parameters_list)
                #arravg = np.array(parameters_avg)
                #print("federated_avg parameters: ", arravg)
                #features = np.round(features, decimals=3)  # Round to 1 decimal
                #print("federated features: ", features)

                #data_dir = "../../../../data"
                #device_id = "client_005"
                #data_gen = store_h5(data_dir,device_id)

                self.save_features(
                    features=params_array,
                    labels=labels_array
                    #metadata={}
                )

            success = True  # Let's assume the update is successful for this example


        except Exception as e:
            print("Error during UploadModelParams:", e)



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

            print("更新前:", dict(f.attrs))  # 检查原有属性
            f.attrs.update(metadata)
            print("更新后:", dict(f.attrs))  # 确认更新结果
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
if __name__ == '__main__':

    data_dir='..\..\..\data'
    device_id='client_004'
    data_saver=DataSaver(data_dir,device_id)
    # 使用示例
    #features = np.random.rand(100, 64).astype(np.float32)  # 模擬ESP32輸出
    labels = np.random.randint(0, 3, size=100)

    #data_saver.save_features(
    #    features=features,
    #    labels=labels
    #)
    features_labls = np.random.rand(100, 65).astype(np.float32)
    for i in range(0, 100, 1):
        features_labls[i][0]=labels[i]
    data_saver.UploadModelParams(request_values=features_labls)
