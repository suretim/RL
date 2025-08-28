import numpy as np
import tensorflow as tf
from tensorflow import keras
from pathlib import Path
import h5py

from flwr.client import NumPyClient

# 加载简单模型
def create_model() -> tf.keras.Model:
    model = keras.Sequential([
        tf.keras.layers.Dense(32, activation='relu', input_shape=(64,)),
        tf.keras.layers.Dense(3, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

gmodel = create_model()

# 获取初始参数
initial_params = [param.numpy() for param in gmodel.trainable_variables]
def normalize_features(features: np.ndarray) -> np.ndarray:
    """標準化特徵數據 (每個維度0均值1方差)"""
    mean = np.mean(features, axis=0)
    std = np.std(features, axis=0)
    return (features - mean) / (std + 1e-7)  # 避免除零


def load_and_preprocess_esp32_data(data_dir: str,
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
    features = normalize_features(features)
    labels = labels.astype(np.int32)

    # 打亂數據
    shuffle_idx = np.random.permutation(len(features))
    features = features[shuffle_idx]
    labels = labels[shuffle_idx]

    return features, labels


from concurrent.futures import ThreadPoolExecutor
def parallel_load(files, max_workers=4):
    """並行加載多個文件"""
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(
            lambda f: load_and_preprocess_esp32_data(f.parent, f.stem.split("_")[0]),
            files
        ))

    features = np.concatenate([r[0] for r in results])
    labels = np.concatenate([r[1] for r in results])
    return features, labels

class ESP32Client(NumPyClient):
    def __init__(self, device_id, data_dir,model):
        self.device_id = device_id
        self.data_dir = data_dir
        self.model = model
        self.param_cache = None  # 参数缓存

    def get_parameters(self, config=None):
        """带缓存的参数获取"""
        if config is None:
            config = {}

        # 使用缓存避免重复计算
        if self.param_cache is not None and not config.get("force_update", False):
            return self.param_cache

        # 获取当前参数
        params = [param.numpy() for param in self.model.trainable_variables]

        # 应用压缩 (如果配置)
        #if config.get("compress", False):
        #    params = self._compress_parameters(params, config)

        # 更新缓存
        self.param_cache = params
        return params
    def load_data(self):
        # 查找該設備的所有數據文件
        device_files = list(Path(self.data_dir).glob(f"{self.device_id}_*.h5"))

        if not device_files:
            raise FileNotFoundError(f"找不到設備 {self.device_id} 的數據")

        # 並行加載數據
        features, labels = parallel_load(device_files)
        return features, labels


    def fit(self, parameters, config):
        """联邦学习客户端训练方法"""
        # 1. 设置模型参数

        self.model.set_weights(parameters)

        # 2. 加载数据
        features, labels = self.load_data()

        # 3. 从配置获取超参数
        batch_size = config.get("batch_size", 32)
        local_epochs = config.get("local_epochs", 1)
        learning_rate = config.get("lr", 0.001)

        # 4. 创建TensorFlow Dataset
        dataset = tf.data.Dataset.from_tensor_slices((features, labels))
        dataset = dataset.shuffle(len(features)).batch(batch_size)

        # 5. 配置优化器和损失函数
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()

        # 6. 训练指标跟踪
        train_loss = tf.keras.metrics.Mean(name="train_loss")
        train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name="train_accuracy")

        # 7. 训练循环
        @tf.function  # 使用图执行加速
        def train_step(x_batch, y_batch):
            with tf.GradientTape() as tape:
                predictions = self.model(x_batch, training=True)
                loss = loss_fn(y_batch, predictions)

            gradients = tape.gradient(loss, self.model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

            train_loss(loss)
            train_accuracy(y_batch, predictions)
            return loss

        print(f"\n客户端 {self.device_id} 开始训练, 样本数: {len(features)}")
        for epoch in range(local_epochs):
            # 重置指标
            train_loss.reset_states()
            train_accuracy.reset_states()

            for x_batch, y_batch in dataset:
                train_step(x_batch, y_batch)

            # 打印训练进度
            print(f"Epoch {epoch + 1}/{local_epochs} - "
                  f"Loss: {train_loss.result():.4f} - "
                  f"Accuracy: {train_accuracy.result():.4f}")

        # 8. 返回更新后的参数和指标
        updated_params = self.model.get_weights()
        num_examples = len(features)
        metrics = {
            "train_loss": float(train_loss.result()),
            "train_accuracy": float(train_accuracy.result())
        }

        return updated_params, num_examples, metrics


base_config = {
    "noise_scale":1.0,
    "batch_size": 32,           # 批量大小
    "local_epochs": 5,          # 本地训练轮数
    "learning_rate": 0.01,      # 学习率
    "optimizer": "adam",        # 优化器类型
    "shuffle": True             # 是否打乱数据
}


# 添加高斯噪声
noisy_params = [
    p + np.random.normal(0, base_config['noise_scale'], p.shape)
    for p in initial_params
]
# 使用示例
client = ESP32Client(device_id="esp32_001", data_dir="data",model= gmodel )
updated_params, num_examples, metrics=client.fit(initial_params, config=base_config)
