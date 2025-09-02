
import os
import glob
import pandas as pd
import numpy as np
from typing import Tuple

 
import tensorflow as tf

#NUM_TASKS = 5
SUPPORT_SIZE = 10
QUERY_SIZE = 20
# ---------- 1. Encoder ----------
def build_hvac_encoder(seq_len=20, n_features=3, latent_dim=64):
    inp = tf.keras.Input(shape=(seq_len, n_features))
    x = tf.keras.layers.LSTM(64, return_sequences=True)(inp)
    x = tf.keras.layers.LSTM(64)(x)
    x = tf.keras.layers.Dense(latent_dim)(x)
    x = tf.math.l2_normalize(x, axis=-1)
    return tf.keras.Model(inp, x, name="encoder")

# ---------- 2. Prototype Classifier ----------
class PrototypeClassifier(tf.keras.Model):
    def __init__(self, n_classes, latent_dim, tau=0.1):
        super().__init__()
        self.prototypes = tf.Variable(tf.random.normal([n_classes, latent_dim]), trainable=False)
        self.tau = tau

    def call(self, z):
        z = tf.math.l2_normalize(z, axis=-1)
        protos = tf.math.l2_normalize(self.prototypes, axis=-1)
        sim = tf.matmul(z, protos, transpose_b=True) / self.tau
        return tf.nn.softmax(sim, axis=-1)

    def update_prototypes(self, features, labels):
        for k in range(self.prototypes.shape[0]):
            mask = tf.where(labels == k)
            if tf.size(mask) > 0:
                mean_vec = tf.reduce_mean(tf.gather(features, mask[:,0]), axis=0)
                self.prototypes[k].assign(mean_vec)

# ---------- 3. VPD 计算 ----------
def calc_vpd(temp, rh):
    es = 0.6108 * np.exp(17.27*temp / (temp+237.3))
    ea = es * rh
    return es - ea

# ---------- 4. 环境类 ----------
class PlantHVACEnv:
    def __init__(self, seq_len=20, n_features=3, temp_init=25.0, humid_init=0.5, latent_dim=64):
        self.seq_len = seq_len
        self.temp_init = temp_init
        self.humid_init = humid_init
        self.encoder = build_encoder(seq_len, n_features, latent_dim)
        self.proto_cls = PrototypeClassifier(n_classes=3, latent_dim=latent_dim)  # seedling, veg, flower
        self.reset()

    def reset(self):
        self.temp = self.temp_init
        self.humid = self.humid_init
        self.health = 0
        self.t = 0
        self.prev_action = np.zeros(4, dtype=int)
        return self._get_state()

    def _get_state(self):
        return np.array([self.health, self.temp, self.humid], dtype=np.float32)

    def step(self, action, seq_input, params):
        """
        action: [AC, Humidifier, Heater, Dehumidifier]
        seq_input: [1, seq_len, n_features] 当前时序序列
        """
        ac, humi, heat, dehumi = action

        # --- 环境动力学 ---
        self.temp += (-0.5 if ac==1 else 0.2) + (0.5 if heat==1 else 0.0)
        self.humid += (0.05 if humi==1 else -0.02) + (-0.03 if heat==1 else 0.0) + (-0.05 if dehumi==1 else 0.0)
        self.temp = np.clip(self.temp, 15, 35)
        self.humid = np.clip(self.humid, 0, 1)

        # 健康判定
        self.health = 0 if 22<=self.temp<=28 and 0.4<=self.humid<=0.7 else 1

        # --- latent soft label ---
        z = self.encoder(seq_input, training=False)
        soft_label = self.proto_cls(z).numpy()[0]  # [seedling, veg, flower]
        flower_prob = soft_label[2]

        # --- reward ---
        # 基础 health + 能耗 + 开关惩罚
        health_reward = 1.0 if self.health==0 else -1.0
        energy_cost = params.get("energy_penalty",0.1) * np.sum(action)
        switch_penalty = params.get("switch_penalty_per_toggle",0.2) * np.sum(np.abs(action - self.prev_action))

        # 花期强化 VPD 控制
        vpd_target = params.get("vpd_target", 1.2)
        vpd_current = calc_vpd(self.temp, self.humid)
        vpd_reward = -abs(vpd_current - vpd_target) * params.get("vpd_penalty",2.0)

        reward = health_reward - energy_cost - switch_penalty + flower_prob * vpd_reward

        self.prev_action = action
        self.t += 1
        done = self.t >= self.seq_len

        info = {
            "latent_soft_label": soft_label,
            "flower_prob": flower_prob,
            "temp": self.temp,
            "humid": self.humid,
            "vpd": vpd_current
        }

        return self._get_state(), reward, done, info
 
 

def compute_scores(temp, humid):
    """计算 temp/ humid / vpd 三个健康分数"""
    # 温度奖励 (目标 25±3)
    temp_penalty = np.abs(np.clip(temp - 25.0, -3, 3)) / 3.0
    temp_score = 1.0 - temp_penalty

    # 湿度奖励 (目标 0.55±0.15)
    humid_penalty = np.abs(np.clip(humid - 0.55, -0.15, 0.15)) / 0.15
    humid_score = 1.0 - humid_penalty

    # VPD 奖励 (目标 0.5~1.2，中心 0.85)
    vpd = compute_vpd(temp, humid)
    vpd_penalty = np.abs(np.clip(vpd - 0.85, -0.35, 0.35)) / 0.35
    vpd_score = 1.0 - vpd_penalty

    # 综合健康度
    health_score = (temp_score + humid_score + vpd_score) / 3.0
    return health_score, temp_score, humid_score, vpd_score, vpd

 

# Q-learning 智能体
# -------------------------------
class QLearningAgent:
    def __init__(self, n_actions=4, alpha=0.1, gamma=0.9, epsilon=0.2):
        self.n_actions = n_actions
        self.q_table = {}
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    def _state_to_key(self, state):
        return tuple(state.tolist())

    def select_action(self, state):
        key = self._state_to_key(state)
        if key not in self.q_table:
            self.q_table[key] = np.zeros(self.n_actions)
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)
        return np.argmax(self.q_table[key])

    def update(self, state, action, reward, next_state, done):
        key = self._state_to_key(state)
        if key not in self.q_table:
            self.q_table[key] = np.zeros(self.n_actions)
        if done:
            target = reward
        else:
            next_key = self._state_to_key(next_state)
            if next_key not in self.q_table:
                self.q_table[next_key] = np.zeros(self.n_actions)
            target = reward + self.gamma * np.max(self.q_table[next_key])
        self.q_table[key][action] += self.alpha * (target - self.q_table[key][action])

def generate_plant_sequence_easy(save_dir,seq_len , noise_std , insect_prob=0.3, equip_fail_prob=0.2):
    t, h, l, labels = [], [], [], []
    ac, heater, dehum, hum = [], [], [], []
    os.makedirs(save_dir, exist_ok=True)
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
        flip_prob = 0.2  # 5% 的概率翻转开关


        ac_state     = ac_state    ^ (np.random.rand() < flip_prob)
        heater_state =heater_state ^ (np.random.rand() < flip_prob)
        hum_state    =hum_state    ^ (np.random.rand() < flip_prob)
        dehum_state  =dehum_state  ^ (np.random.rand() < flip_prob)
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



def generate_plant_sequence(save_dir, seq_len, noise_std, insect_prob=0.5, equip_fail_prob=0.5):
    """
    生成植物生长序列数据

    参数:
    - save_dir: 保存目录
    - seq_len: 序列长度
    - noise_std: 噪声标准差
    - insect_prob: 虫害发生概率
    - equip_fail_prob: 设备故障概率

    返回:
    - 包含植物生长数据的DataFrame
    """
    # 初始化数据列表
    temperature, humidity, light, labels = [], [], [], []
    ac, heater, dehum, humidifier = [], [], [], []

    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)

    # 随机虫害事件
    insect_event = np.random.rand() < insect_prob
    insect_start = np.random.randint(300, 800) if insect_event else -1
    insect_end = insect_start + np.random.randint(50, 150) if insect_event else -1

    # 随机设备异常事件
    equip_fail_event = np.random.rand() < equip_fail_prob
    fail_type = None
    fail_start, fail_end = -1, -1

    if equip_fail_event:
        fail_type = np.random.choice(["humidifier_fail", "dehumidifier_fail", "heater_fail", "ac_fail"])
        fail_start = np.random.randint(200, 700)
        fail_end = fail_start + np.random.randint(80, 200)

    # 设备状态翻转概率
    flip_prob = 0.05  # 降低翻转概率，使异常更明显

    for step in range(seq_len):
        # ========= 生命周期阶段 =========
        if step < 200:  # 育苗期
            base_t, base_h, base_l = 22, 65, 250
        elif step < 600:  # 生长期
            base_t, base_h, base_l = 25, 58, 400
        else:  # 开花期
            base_t, base_h, base_l = 28, 48, 600

        # 基础波动 + 噪声
        ti = base_t + np.sin(step / 50) + np.random.randn() * noise_std
        hi = base_h + np.cos(step / 70) + np.random.randn() * noise_std
        li = base_l + np.sin(step / 100) * 20 + np.random.randn() * noise_std * 5

        # 初始化标签和HVAC状态
        label = 0  # 默认为健康
        ac_state, heater_state, dehum_state, hum_state = 0, 0, 0, 0

        # ========= 虫害事件 =========
        if insect_event and insect_start <= step <= insect_end:
            li *= np.random.uniform(0.6, 0.8)  # 光照下降
            hi += np.random.uniform(-5, 5)  # 湿度异常波动
            label = 2  # 标记为不健康

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

        # ========= 健康状态检查 (如果没有被异常事件标记) =========
        if label == 0:
            if (ti < 10) or (li < 100):
                label = 1  # 非植物状态
            elif (ti < 15) or (ti > 35) or (hi < 30) or (hi > 80) or (li > 800):
                label = 2  # 不健康

        # ========= HVAC 正常控制逻辑 (如果没有设备故障) =========
        if not (equip_fail_event and fail_start <= step <= fail_end):
            ac_state = 1 if ti > 26 else 0
            heater_state = 1 if ti < 20 else 0
            dehum_state = 1 if hi > 70 else 0
            hum_state = 1 if hi < 40 else 0

        # ========= 异常期间的随机扰动 =========
        if label == 2:
            if np.random.rand() < 0.15:  # 增加异常期间的扰动概率
                ac_state = 1 - ac_state
            if np.random.rand() < 0.15:
                heater_state = 1 - heater_state
            if np.random.rand() < 0.15:
                dehum_state = 1 - dehum_state
            if np.random.rand() < 0.15:
                hum_state = 1 - hum_state
        else:
            # 正常状态下的随机翻转
            ac_state = ac_state ^ (np.random.rand() < flip_prob)
            heater_state = heater_state ^ (np.random.rand() < flip_prob)
            hum_state = hum_state ^ (np.random.rand() < flip_prob)
            dehum_state = dehum_state ^ (np.random.rand() < flip_prob)

        # 存储数据
        temperature.append(ti)
        humidity.append(hi)
        light.append(li)
        labels.append(label)
        ac.append(ac_state)
        heater.append(heater_state)
        dehum.append(dehum_state)
        humidifier.append(hum_state)

    # 创建DataFrame
    df = pd.DataFrame({
        "temp": temperature,
        "humid": humidity,
        "light": light,
        "ac": ac,
        "heater": heater,
        "dehum": dehum,
        "hum": humidifier,
        "label": labels
    })

    return df
# ------------------ 载入 CSV 预训练 encoder ------------------
def load_all_csvs(data_dir):
    dfs = [pd.read_csv(os.path.join(data_dir, f)) for f in os.listdir(data_dir) if f.endswith(".csv")]
    return pd.concat(dfs, axis=0).reset_index(drop=True)

def load_csvs(data_dir,seq_len,num_features):
    X_labeled_list, y_labeled_list, X_unlabeled_list = [], [], []
    files = sorted(glob.glob(os.path.join(data_dir,"*.csv")))
    for f in files:
        df = pd.read_csv(f).fillna(-1)
        data = df.values.astype(np.float32)
        feats, labels = data[:,:-1], data[:,-1]
        for i in range(len(data)-seq_len+1):
            w_x = feats[i:i+seq_len]
            if w_x.shape != (seq_len, num_features):
                continue  # 跳过不完整窗口
            w_y = labels[i+seq_len-1]
            if w_y==-1:
                X_unlabeled_list.append(w_x)
            else:
                X_labeled_list.append(w_x)
                y_labeled_list.append(int(w_y))
    X_unlabeled = np.array(X_unlabeled_list) if X_unlabeled_list else np.random.randn(200,seq_len,num_features).astype(np.float32)
    X_labeled = np.array(X_labeled_list) if X_labeled_list else np.empty((0,seq_len,num_features),dtype=np.float32)
    y_labeled = np.array(y_labeled_list) if y_labeled_list else np.empty((0,),dtype=np.int32)
    return X_unlabeled, X_labeled, y_labeled


#,seq_len=1000
def load_csv_data(load_dir: str, seq_len) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Load CSVs and build sliding windows.
    Returns: X_unlabeled, X_labeled, y_labeled, num_feats
    """
    X_labeled_list, y_labeled_list, X_unlabeled_list = [], [], []

    files = sorted(glob.glob(load_dir))
    if not files:
        print(f"⚠️ No CSV files matched: {load_dir}. Using random fallback for unlabeled pretraining.")

    for file in files:
        df = pd.read_csv(file).fillna(-1)
        data = df.values.astype(np.float32)
        feats, labels = data[:, :-1], data[:, -1]
        for i in range(len(data) - seq_len + 1):
            w_x = feats[i:i + seq_len]
            w_y = labels[i + seq_len - 1]
            if w_y == -1:
                X_unlabeled_list.append(w_x)
            else:
                X_labeled_list.append(w_x)
                y_labeled_list.append(int(w_y))

    X_unlabeled = np.array(X_unlabeled_list, dtype=np.float32) if len(X_unlabeled_list) > 0 else np.empty((0,), dtype=np.float32)

    if len(X_labeled_list) > 0:
        X_labeled = np.array(X_labeled_list, dtype=np.float32)
        y_labeled = np.array(y_labeled_list, dtype=np.int32)
    else:
        X_labeled = np.empty((0, seq_len, X_unlabeled.shape[2] if X_unlabeled.size > 0 else 7), dtype=np.float32)
        y_labeled = np.empty((0,), dtype=np.int32)

    num_feats = X_labeled.shape[2] if X_labeled.size > 0 else (X_unlabeled.shape[2] if X_unlabeled.size > 0 else 7)

    if num_feats < 7:
        raise ValueError(
            "Expected at least 7 features per timestep: [temp, humid, light, ac, heater, dehum, hum]. Found: %d" % num_feats
        )

    if X_unlabeled.size == 0:
        # provide unlabeled fallback if none
        X_unlabeled = np.random.randn(200, seq_len, num_feats).astype(np.float32)

    return X_unlabeled, X_labeled, y_labeled, num_feats


def sample_tasks(X: np.ndarray, y: np.ndarray, num_tasks ,
                 support_size: int = SUPPORT_SIZE, query_size: int = QUERY_SIZE):
    tasks = []
    n = len(X)
    if n < support_size + query_size:
        #print(n,support_size, query_size)
        return tasks
        #raise ValueError(f"Not enough labeled samples to build tasks: need {support_size+query_size}, got {n}")
    for _ in range(num_tasks):
        idx = np.random.choice(n, support_size + query_size, replace=False)
        X_support, y_support = X[idx[:support_size]], y[idx[:support_size]]
        X_query, y_query = X[idx[support_size:]], y[idx[support_size:]]
        tasks.append((X_support, y_support, X_query, y_query))
    return tasks


class TFLiteModelWrapper:
    def __init__(self, tflite_model_path):
        self.interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def predict(self, x):
        # Ensure input is float32
        x = np.array(x, dtype=np.float32)
        self.interpreter.set_tensor(self.input_details[0]['index'], x)
        self.interpreter.invoke()
        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
        return output_data

