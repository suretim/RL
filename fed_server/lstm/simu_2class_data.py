import os
import numpy as np
import pandas as pd

# ==== 参数 ====
SAVE_DIR = "./data"
os.makedirs(SAVE_DIR, exist_ok=True)
NUM_FILES = 50       # 生成多少个 CSV 文件
SEQ_LEN = 500       # 每个文件的长度
NUM_CLASSES = 3     # 分类标签数量，0/1
NOISE_STD = 0.05    # 模拟噪声大小

# ==== 随机生成传感器数据 ====
for i in range(NUM_FILES):
    # 模拟温度、湿度、光照
    t = 20 + 5 * np.sin(np.linspace(0, 10, SEQ_LEN)) + np.random.randn(SEQ_LEN) * NOISE_STD
    h = 50 + 10 * np.cos(np.linspace(0, 5, SEQ_LEN)) + np.random.randn(SEQ_LEN) * NOISE_STD
    l = 300 + 50 * np.sin(np.linspace(0, 3, SEQ_LEN)) + np.random.randn(SEQ_LEN) * NOISE_STD

    # 简单生成标签：假设 temp > 22 就标 1，否则 0（仅作示例）
    label = (t > 22).astype(int)

    df = pd.DataFrame({
        "temp": t,
        "humid": h,
        "light": l,
        "label": label
    })

    file_path = os.path.join(SAVE_DIR, f"sensor_data_{i}.csv")
    df.to_csv(file_path, index=False)
    print(f"Saved {file_path}, shape: {df.shape}")
