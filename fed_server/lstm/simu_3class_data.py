import os
import numpy as np
import pandas as pd

SAVE_DIR = "../../../../data/lll_data"
os.makedirs(SAVE_DIR, exist_ok=True)
NUM_FILES = 50        # 生成几个时序文件
SEQ_LEN = 1000       # 每个文件的长度
NOISE_STD = 0.5      # 噪声强度

def generate_plant_sequence(seq_len=1000, noise_std=0.5, insect_prob=0.3):
    t, h, l, labels = [], [], [], []

    # 随机选择是否引入虫害
    insect_event = np.random.rand() < insect_prob
    insect_start = np.random.randint(300, 800) if insect_event else -1
    insect_end   = insect_start + np.random.randint(50, 150) if insect_event else -1

    for step in range(seq_len):
        # ========= 生命周期阶段 =========
        if step < 200:   # 育苗期
            base_t, base_h, base_l = 22, 65, 250
        elif step < 600: # 生长期
            base_t, base_h, base_l = 25, 58, 400
        else:            # 开花期
            base_t, base_h, base_l = 28, 48, 600

        # 基础波动 + 噪声
        ti = base_t + np.sin(step/50) + np.random.randn() * noise_std
        hi = base_h + np.cos(step/70) + np.random.randn() * noise_std
        li = base_l + np.sin(step/100) * 20 + np.random.randn() * noise_std * 5

        # ========= 虫害事件 =========
        if insect_event and insect_start <= step <= insect_end:
            li *= np.random.uniform(0.6, 0.8)  # 光照下降
            hi += np.random.uniform(-5, 5)     # 湿度异常波动
            label = 2   # 虫害也标为 "不健康"
        else:
            # ========= 标签 =========
            if (ti < 10) or (li < 100):
                label = 1  # 非植物
            elif (ti < 15) or (ti > 35) or (hi < 30) or (hi > 80) or (li > 800):
                label = 2  # 不健康
            else:
                label = 0  # 健康

        t.append(ti)
        h.append(hi)
        l.append(li)
        labels.append(label)

    return pd.DataFrame({"temp": t, "humid": h, "light": l, "label": labels})

# ==== 批量生成 ====
for i in range(NUM_FILES):
    df = generate_plant_sequence(SEQ_LEN, NOISE_STD)
    file_path = os.path.join(SAVE_DIR, f"plant_seq_with_insect_{i}.csv")
    df.to_csv(file_path, index=False)
    print(f"Saved {file_path}, shape: {df.shape}")
