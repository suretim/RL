import os
import numpy as np
import pandas as pd

SAVE_DIR = "../../../../data/lll_data"
os.makedirs(SAVE_DIR, exist_ok=True)
NUM_FILES = 50
SEQ_LEN = 1000
NOISE_STD = 0.5

def generate_plant_sequence(seq_len=1000, noise_std=0.5, insect_prob=0.3, equip_fail_prob=0.2):
    t, h, l, labels = [], [], [], []
    ac, heater, dehum, hum = [], [], [], []

    # 随机虫害事件
    insect_event = np.random.rand() < insect_prob
    insect_start = np.random.randint(300, 800) if insect_event else -1
    insect_end   = insect_start + np.random.randint(50, 150) if insect_event else -1

    # 随机设备异常事件
    equip_fail_event = np.random.rand() < equip_fail_prob
    fail_type = None
    fail_start, fail_end = -1, -1
    if equip_fail_event:
        fail_type = np.random.choice(["humidifier_fail", "dehumidifier_fail", "heater_fail", "ac_fail"])
        fail_start = np.random.randint(200, 700)
        fail_end = fail_start + np.random.randint(80, 200)

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
            label = 2
        else:
            # ========= 默认标签 =========
            if (ti < 10) or (li < 100):
                label = 1  # 非植物
            elif (ti < 15) or (ti > 35) or (hi < 30) or (hi > 80) or (li > 800):
                label = 2  # 不健康
            else:
                label = 0  # 健康

        # ========= 设备异常事件 =========
        if equip_fail_event and fail_start <= step <= fail_end:
            if fail_type == "humidifier_fail":
                hum_state = 1  # 一直开
                hi += np.random.uniform(5, 15)  # 湿度过重
                label = 2
            elif fail_type == "dehumidifier_fail":
                dehum_state = 1
                hi -= np.random.uniform(5, 15)  # 湿度过低
                label = 2
            elif fail_type == "heater_fail":
                heater_state = 1
                ti += np.random.uniform(5, 10)  # 过热
                label = 2
            elif fail_type == "ac_fail":
                ac_state = 1
                ti -= np.random.uniform(5, 10)  # 过冷
                label = 2

        # ========= HVAC 正常控制逻辑 =========
        ac_state = 1 if ti > 26 else 0
        heater_state = 1 if ti < 20 else 0
        dehum_state = 1 if hi > 70 else 0
        hum_state = 1 if hi < 40 else 0

        # ========= 异常/虫害期间扰动 =========
        if label == 2:
            if np.random.rand() < 0.1: ac_state = 1 - ac_state
            if np.random.rand() < 0.1: heater_state = 1 - heater_state
            if np.random.rand() < 0.1: dehum_state = 1 - dehum_state
            if np.random.rand() < 0.1: hum_state = 1 - hum_state

        t.append(ti)
        h.append(hi)
        l.append(li)
        labels.append(label)
        ac.append(ac_state)
        heater.append(heater_state)
        dehum.append(dehum_state)
        hum.append(hum_state)

    return pd.DataFrame({
        "temp": t,
        "humid": h,
        "light": l,
        "ac": ac,
        "heater": heater,
        "dehum": dehum,
        "hum": hum,
        "label": labels
    })

# ==== 批量生成 ====
for i in range(NUM_FILES):
    df = generate_plant_sequence(SEQ_LEN, NOISE_STD)
    file_path = os.path.join(SAVE_DIR, f"plant_seq_with_hvac_fail_{i}.csv")
    df.to_csv(file_path, index=False)
    print(f"Saved {file_path}, shape: {df.shape}")
