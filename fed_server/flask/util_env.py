
import gym
from gym import spaces

import numpy as np
import tensorflow as tf
from collections import deque


class PlantHVACEnv(gym.Env):
    """
    自定義 Plant HVAC 環境
    狀態 (state_dim=5):
        [溫度, 濕度, 光照, CO2濃度, 土壤濕度]
    動作 (action_dim=4):
        0 = 空調降溫
        1 = 加濕器增濕
        2 = 開燈補光
        3 = 通風降CO2

    獎勵:
        根據與最佳生長區間的差距給分
    """
    def __init__(self):
        super(PlantHVACEnv, self).__init__()

        # 觀測空間: 連續狀態 (5維)
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0, 200.0, 0.0], dtype=np.float32),
            high=np.array([50.0, 100.0, 2000.0, 2000.0, 100.0], dtype=np.float32),
            dtype=np.float32
        )

        # 動作空間: 4個離散動作
        self.action_space = spaces.Discrete(4)

        # 初始狀態
        self.state = None
        self.reset()

        # 最佳生長區間 (目標區域)
        self.optimal_ranges = {
            "temp": (22, 28),      # 溫度 (°C)
            "humidity": (60, 75),  # 濕度 (%)
            "light": (400, 800),   # 光照 (lux)
            "co2": (350, 600),     # CO2 ppm
            "soil": (40, 60),      # 土壤濕度 (%)
        }

    def reset(self):
        """重置環境"""
        self.state = self.observation_space.sample()
        return self.state

    def step(self, action):
        """
        根據 action 更新狀態
        """
        temp, humidity, light, co2, soil = self.state

        # 簡單模擬 HVAC 控制
        if action == 0:   # 降溫
            temp -= 1.5
        elif action == 1: # 加濕
            humidity += 2.0
        elif action == 2: # 補光
            light += 50
        elif action == 3: # 通風
            co2 -= 30

        # 加入隨機擾動 (模擬外部環境影響)
        temp += np.random.uniform(-0.5, 0.5)
        humidity += np.random.uniform(-1, 1)
        light += np.random.uniform(-20, 20)
        co2 += np.random.uniform(-10, 10)
        soil += np.random.uniform(-1, 1)

        # 更新狀態
        self.state = np.array([temp, humidity, light, co2, soil], dtype=np.float32)

        # 計算 reward
        reward = self._calculate_reward()

        # 終止條件：偏離太嚴重
        done = self._is_done()

        return self.state, reward, done, {}

    def _calculate_reward(self):
        """根據與最佳生長區間的差距計算 reward"""
        temp, humidity, light, co2, soil = self.state
        score = 0

        def range_penalty(value, optimal_range):
            low, high = optimal_range
            if value < low:
                return -(low - value)
            elif value > high:
                return -(value - high)
            else:
                return +1.0  # 在區間內加分

        score += range_penalty(temp, self.optimal_ranges["temp"])
        score += range_penalty(humidity, self.optimal_ranges["humidity"])
        score += range_penalty(light, self.optimal_ranges["light"])
        score += range_penalty(co2, self.optimal_ranges["co2"])
        score += range_penalty(soil, self.optimal_ranges["soil"])

        return score

    def _is_done(self):
        """當狀態超出極限值，結束 episode"""
        temp, humidity, light, co2, soil = self.state
        if temp < 0 or temp > 50:
            return True
        if humidity < 0 or humidity > 100:
            return True
        if light < 0 or light > 2000:
            return True
        if co2 < 100 or co2 > 3000:
            return True
        if soil < 0 or soil > 100:
            return True
        return False

class PlantLLLHVACEnv:
    def __init__(self, seq_len=10, n_features=5, temp_init=25.0, humid_init=0.5,
                 latent_dim=64, mode="growing"):
        self.seq_len = seq_len
        self.temp_init = temp_init
        self.humid_init = humid_init
        self.n_features = n_features  # 現在有5個特徵: temp, humid, health, light, co2
        self.mode = mode  # "growing", "flowering", "seeding"

        # 構建encoder
        self.encoder = self._build_encoder(seq_len, n_features, latent_dim)

        # LLL模型
        self.lll_model = self._build_lll_model(latent_dim, hidden_dim=64, output_dim=3)
        self.fisher_matrix = None
        self.prev_weights = None
        self.memory = deque(maxlen=1000)  # 簡單的記憶緩衝區

        # 不同模式的理想環境參數（添加光照和CO2範圍）
        self.mode_params = {
            "growing": {
                "temp_range": (22, 28),
                "humid_range": (0.4, 0.7),
                "vpd_range": (0.8, 1.5),
                "light_range": (300, 600),  # lux
                "co2_range": (400, 800)  # ppm
            },
            "flowering": {
                "temp_range": (20, 26),
                "humid_range": (0.4, 0.6),
                "vpd_range": (1.0, 1.8),
                "light_range": (500, 800),  # lux
                "co2_range": (600, 1000)  # ppm
            },
            "seeding": {
                "temp_range": (24, 30),
                "humid_range": (0.5, 0.7),
                "vpd_range": (0.7, 1.3),
                "light_range": (200, 400),  # lux
                "co2_range": (400, 600)  # ppm
            }
        }

        # 初始化環境變量
        self.light = 500.0  # 初始光照 (lux)
        self.co2 = 600.0  # 初始CO2濃度 (ppm)

        # 初始化狀態變量
        self.reset()

    def _build_encoder(self, seq_len, n_features, latent_dim):
        """構建序列編碼器"""
        return tf.keras.Sequential([
            tf.keras.layers.LSTM(64, input_shape=(seq_len, n_features), return_sequences=True),
            tf.keras.layers.LSTM(32),
            tf.keras.layers.Dense(latent_dim, activation='relu')
        ])

    def _build_lll_model(self, input_dim, hidden_dim, output_dim):
        """構建終身學習模型"""
        return tf.keras.Sequential([
            tf.keras.layers.Dense(hidden_dim, activation='relu', input_shape=(input_dim,)),
            tf.keras.layers.Dense(hidden_dim, activation='relu'),
            tf.keras.layers.Dense(output_dim, activation='softmax')
        ])

    def update_lll_model(self, sequence_input, true_label=None):
        """
        更新LLL模型並返回預測結果

        Args:
            sequence_input: 輸入序列數據
            true_label: 真實標籤（可選，用於訓練）

        Returns:
            soft_label: 軟標籤預測概率
        """
        # 編碼序列數據
        latent_representation = self.encoder(sequence_input)

        # 獲取預測
        soft_label = self.lll_model(latent_representation)

        # 如果有真實標籤，則進行訓練
        if true_label is not None:
            self._train_lll_model(latent_representation, true_label)

        return soft_label.numpy()[0]  # 返回第一個batch的預測

    def _train_lll_model(self, latent_input, true_label):
        """
        訓練LLL模型
        """
        # 計算EWC正則化損失
        ewc_loss = self._compute_ewc_loss()

        with tf.GradientTape() as tape:
            predictions = self.lll_model(latent_input, training=True)

            # 計算交叉熵損失
            ce_loss = tf.keras.losses.sparse_categorical_crossentropy(
                true_label, predictions
            )

            # 總損失 = 交叉熵損失 + EWC正則化
            total_loss = tf.reduce_mean(ce_loss) + ewc_loss

        # 更新模型權重
        gradients = tape.gradient(total_loss, self.lll_model.trainable_variables)
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        optimizer.apply_gradients(zip(gradients, self.lll_model.trainable_variables))

    def _compute_ewc_loss(self):
        """計算EWC正則化損失"""
        if self.fisher_matrix is None or self.prev_weights is None:
            return 0.0

        ewc_loss = 0.0
        ewc_lambda = 500  # EWC正則化強度

        for i, (current_var, prev_var, fisher) in enumerate(
                zip(self.lll_model.trainable_variables,
                    self.prev_weights,
                    self.fisher_matrix)):
            ewc_loss += tf.reduce_sum(
                ewc_lambda * 0.5 * fisher * tf.square(current_var - prev_var)
            )

        return ewc_loss

    def save_model_knowledge(self):
        """保存當前模型的知識（用於EWC）"""
        # 保存當前權重
        self.prev_weights = [tf.identity(var) for var in self.lll_model.trainable_variables]

        # 計算Fisher信息矩陣（這裡簡化處理）
        self.fisher_matrix = [
            tf.ones_like(var) * 0.1 for var in self.lll_model.trainable_variables
        ]

    def reset(self):
        """重置環境狀態"""
        self.temp = self.temp_init
        self.humid = self.humid_init
        self.light = 500.0
        self.co2 = 600.0
        self.health = 2  # 初始狀態設為"無法判定"
        self.t = 0
        self.prev_action = np.zeros(4, dtype=int)

        # 初始化序列數據
        self.current_sequence = np.zeros((1, self.seq_len, self.n_features))

        # 填充初始序列
        for i in range(self.seq_len):
            self.current_sequence[0, i] = [self.health,self.temp, self.humid,  self.light, self.co2]

        return self._get_state()

    def _get_state(self):
        """獲取當前狀態"""
        return np.array([self.health, self.temp, self.humid, self.light, self.co2], dtype=np.float32)

    def update_sequence(self, new_data_point):
        """
        更新數據序列，添加新的數據點

        Args:
            new_data_point: 新的傳感器數據點 [temp, humid, health, light, co2]
        """
        # 將序列向前移動一位，移除最舊的數據
        self.current_sequence = np.roll(self.current_sequence, shift=-1, axis=1)

        # 在序列末尾添加新的數據點
        self.current_sequence[0, -1] = new_data_point

    def _calculate_enhanced_vpd(self, temp, humid, light, co2):
        """
        計算增強的VPD，考慮光照和CO2的影響
        """
        # 基礎VPD計算
        base_vpd = self.calc_vpd(temp, humid)

        # 光照對VPD的影響因子
        light_factor = np.clip((light - 200) / 600, 0.8, 1.2)

        # CO2對VPD的影響因子
        co2_factor = np.clip(1.0 - (co2 - 400) / 1000, 0.8, 1.0)

        # 增強的VPD計算
        enhanced_vpd = base_vpd * light_factor * co2_factor

        return enhanced_vpd

    def calc_vpd(self, temp, humid):
        """計算VPD（蒸汽壓差）"""
        # 飽和蒸汽壓計算（Tetens公式）
        es = 0.6108 * np.exp(17.27 * temp / (temp + 237.3))
        # 實際蒸汽壓
        ea = es * humid
        # VPD
        vpd = es - ea
        return max(vpd, 0.1)

    def step(self, action, params=None, true_label=None):
        """
        執行動作並返回新的狀態、獎勵等信息
        """
        if params is None:
            params = {}

        ac, humi, heat, dehumi = action

        # 環境動力學
        self.temp += (-0.5 if ac == 1 else 0.2) + (0.5 if heat == 1 else 0.0)
        self.humid += (0.05 if humi == 1 else -0.02) + (-0.03 if heat == 1 else 0.0) + (-0.05 if dehumi == 1 else 0.0)

        # 光照和CO2的自然變化
        self.light += np.random.normal(0, 20)
        self.co2 += np.random.normal(0, 10)

        # 邊界限制
        self.temp = np.clip(self.temp, 15, 35)
        self.humid = np.clip(self.humid, 0, 1)
        self.light = np.clip(self.light, 100, 1000)
        self.co2 = np.clip(self.co2, 300, 1200)

        # 根據當前模式獲取理想環境參數
        mode_param = self.mode_params[self.mode]
        temp_min, temp_max = mode_param["temp_range"]
        humid_min, humid_max = mode_param["humid_range"]
        vpd_min, vpd_max = mode_param["vpd_range"]
        light_min, light_max = mode_param["light_range"]
        co2_min, co2_max = mode_param["co2_range"]

        # 計算增強的VPD
        vpd_current = self._calculate_enhanced_vpd(self.temp, self.humid, self.light, self.co2)

        # 健康判定
        temp_ok = temp_min <= self.temp <= temp_max
        humid_ok = humid_min <= self.humid <= humid_max
        vpd_ok = vpd_min <= vpd_current <= vpd_max
        light_ok = light_min <= self.light <= light_max
        co2_ok = co2_min <= self.co2 <= co2_max

        # 綜合健康判定
        optimal_conditions = sum([temp_ok, humid_ok, vpd_ok, light_ok, co2_ok])

        if optimal_conditions >= 4:
            self.health = 0  # 健康
        elif optimal_conditions >= 2:
            self.health = 1  # 亞健康
        else:
            self.health = 2  # 不健康

        # 更新序列數據
        new_data_point = np.array([self.health,self.temp, self.humid,  self.light, self.co2])
        self.update_sequence(new_data_point)

        # LLL模型預測
        seq_input_tf = tf.convert_to_tensor(self.current_sequence, dtype=tf.float32)

        if true_label is not None and not isinstance(true_label, tf.Tensor):
            true_label_tf = tf.convert_to_tensor(true_label, dtype=tf.int32)
        else:
            true_label_tf = true_label

        # 使用update_lll_model方法獲取軟標籤
        soft_label = self.update_lll_model(seq_input_tf, true_label_tf)
        flower_prob = soft_label[2]

        # 計算獎勵
        health_reward = {0: 2.0, 1: 0.5, 2: -1.0}[self.health]
        energy_cost = params.get("energy_penalty", 0.1) * np.sum(action)
        switch_penalty = params.get("switch_penalty_per_toggle", 0.2) * np.sum(np.abs(action - self.prev_action))

        # 環境因子獎勵
        vpd_ideal = (vpd_min + vpd_max) / 2
        vpd_reward = -abs(vpd_current - vpd_ideal) * params.get("vpd_penalty", 2.0)

        light_ideal = (light_min + light_max) / 2
        light_reward = -abs(self.light - light_ideal) * params.get("light_penalty", 0.5)

        co2_ideal = (co2_min + co2_max) / 2
        co2_reward = -abs(self.co2 - co2_ideal) * params.get("co2_penalty", 0.3)

        learning_reward = 0
        if true_label is not None:
            pred_class = np.argmax(soft_label)
            true_class = true_label if isinstance(true_label, (int, np.integer)) else true_label.numpy()
            learning_reward = 0.5 if pred_class == true_class else -0.3

        # 軟標籤獎勵
        soft_label_bonus = 0
        if self.mode == "flowering":
            soft_label_bonus = flower_prob * params.get("flower_bonus", 0.5)
        elif self.mode == "seeding":
            soft_label_bonus = soft_label[1] * params.get("seed_bonus", 0.5)
        else:
            soft_label_bonus = soft_label[0] * params.get("grow_bonus", 0.5)

        reward = (health_reward - energy_cost - switch_penalty +
                  vpd_reward + light_reward + co2_reward +
                  learning_reward + soft_label_bonus)

        self.prev_action = action
        self.t += 1
        done = self.t >= self.seq_len

        info = {
            "latent_soft_label": soft_label,
            "flower_prob": flower_prob,
            "temp": self.temp,
            "humid": self.humid,
            "vpd": vpd_current,
            "light": self.light,
            "co2": self.co2,
            "learning_reward": learning_reward,
            "soft_label_bonus": soft_label_bonus,
            "health_status": self.health,
            "health_status_text": ["健康", "亞健康", "不健康"][self.health],
            "optimal_conditions": optimal_conditions
        }

        return self._get_state(), reward, done, info

