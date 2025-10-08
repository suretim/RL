 
import numpy as np
import tensorflow as tf
from collections import deque
import tensorflow_probability as tfp
#from pygments.console import light
import random


# 現在有5個特徵: health,temp, humid,light, co2
# "growing", "flowering", "seeding"
class PlantLLLHVACEnv:
    def __init__(self, seq_len=10,  temp_init=25.0, humid_init=0.60 ,water_init=0.40,light_init=500,co2_init=600,ph_init=7.0, mode="growing",verbose=True):
        self.seq_len = seq_len
        self.temp_init = temp_init
        self.humid_init = humid_init
        self.water_init = water_init
        self.light_init = light_init
        self.co2_init = co2_init
        self.ph_init = ph_init
        #self.prev_action = 0
        self.state_dim = 7  # 7個特徵: health,temp, humid,soil,light, co2,ph
        self.mode = mode   # "growing", "flowering", "seeding"
        self.action_dim = 8 # PORT_CNT
        self.latent_dim = 32
        # 構建encoder
        self.lstm_encoder = self._build_lstm_encoder(seq_len, self.state_dim,self.latent_dim)

        # LLL模型
        self.lll_model = self._build_lll_model(self.latent_dim, hidden_dim=self.latent_dim, output_dim=self.action_dim)
        self.fisher_matrix = None
        self.prev_weights = None
        self.memory = deque(maxlen=1000)  # 簡單的記憶緩衝區

        # 不同模式的理想環境參數（添加光照和CO2範圍）
        self.mode_params = {
            "growing": {
                "temp_range": (22, 28),
                "humid_range": (0.40, 0.70),
                "soil_range": (0.30, 0.50),
                "light_range": (300, 600),  # lux
                "co2_range": (400, 800),  # ppm
                "ph_range": (5.8, 6.5),
                "vpd_range": (0.8, 1.5)
            },
            "flowering": {
                "temp_range": (20, 26),
                "humid_range": (0.40, 0.60),
                "soil_range": (0.30, 0.50),
                "light_range": (500, 800),
                "co2_range": (600, 1000) ,
                "ph_range": (5.8, 6.3),
                "vpd_range": (1.0, 1.8)
            },
            "seeding": {
                "temp_range": (24, 30),
                "humid_range": (0.50, 0.70),
                "soil_range": (0.30, 0.50),
                "light_range": (200, 400),  # lux
                "co2_range": (400, 600)  ,
                "ph_range": (5.5, 6.2),
                "vpd_range": (0.7, 1.3)
            }
        }

        # 初始化狀態變量
        self.reset()
        # === 動作 ===
        self.prev_action = np.zeros(self.action_dim, dtype=float)  # [ac, heat, humid, dehumi, waterpomb, light, carbon, ph]

        # === 內部設備狀態 ===
        self.device_action = {
            "ac": 0.0,
            "heat": 0.0,
            "humid": 0.0,
            "dehumi": 0.0,
            "waterpomb": 0.0,
            "light": 0.0,
            "carbon": 0.0,
            "ph": 0.0,
        }

        self.verbose = verbose
    #input_shape=(seq_len, n_features)
    #output shape=latent_dim
    def _build_lstm_encoder(self, seq_len, n_features, latent_dim):
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
        latent_representation = self.lstm_encoder(sequence_input)

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
        self.water=self.water_init
        self.light = self.light_init
        self.co2 = self.co2_init
        self.ph = self.ph_init

        self.health = 2  # 初始狀態設為"無法判定"
        self.t = 0
        self.prev_action = np.zeros(self.action_dim, dtype=int)

        # 初始化序列數據
        self.current_sequence = np.zeros((1, self.seq_len, self.state_dim))

        # 填充初始序列
        for i in range(self.state_dim):
            self.current_sequence[0, i] = [self.health,self.temp, self.humid,self.water,self.light, self.co2,self.ph]
        #self.__init__(verbose=self.verbose)

        return self._get_state()


    # === 獲取狀態 ===
    def _get_state(self):
        """回傳觀測狀態向量"""
        state = np.array([
            self.health,
            self.temp,
            self.humid,
            self.water,
            self.light,
            self.co2,
            self.ph,
            #*self.prev_action,  # 加入上一動作向量 (可選)
        ], dtype=float)
        return state
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

    def pid_map(self, x, in_min, in_max, out_min, out_max):
        if x < in_min:
            return out_min
        if x > in_max:
            return out_max
        return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min

    def _apply_action(self, action_vector):
        """直接將動作向量應用到設備，不使用 action_mapping"""
        assert len(action_vector) == 8, "Action vector must have 8 elements"
        self.prev_action = np.array(action_vector, dtype=float)

        # 解構動作
        #(
        #    dev_heat,dev_ac,  dev_humid, dev_dehumi,
        #    dev_waterpomb, dev_light, dev_carbon, dev_ph
        #) = self.prev_action

        dev_heat, dev_ac, dev_humid, dev_dehumi, dev_waterpomb, dev_light, dev_carbon, dev_ph=action_vector
        # 更新設備狀態
        self.device_action.update({
            "heat": dev_heat,
            "ac": dev_ac,
            "humid": dev_humid,
            "dehumi": dev_dehumi,
            "waterpomb": dev_waterpomb,
            "light": dev_light,
            "carbon": dev_carbon,
            "ph": dev_ph,
        })

        if self.verbose:
            print(f"[ApplyAction] Device states: {self.device_action}")

    # === 奖励函数 ===
    def _calculate_reward(self, state):
        """根據植物健康與穩定性計算獎勵"""
        temp, humid, light, ph = state[1], state[2], state[3], state[5]
        temp_reward = -abs(temp - 25)
        humid_reward = -abs(humid - 0.6)
        light_reward = -abs(light - 500)
        ph_reward = -abs(ph - 6.5)
        reward = temp_reward + humid_reward + light_reward + ph_reward
        return reward

    # === 判定是否結束 ===
    def _check_done(self, state):
        health = state[0]
        done = health <= 0.2
        if done is False:
            done = self.t >= self.seq_len
        return done  # 植物太不健康就結束

    def xsafe_select_action(self, agent):
        """
        安全选择动作函数，防止 shape 错误和空值
        返回: action, action_prob
        """
        #import numpy as np
        #import tensorflow as tf
        state = np.array([self.health, self.temp, self.humid, self.water, self.light, self.co2, self.ph])

        # --- 确保 state 维度正确 ---
        if state is None or (isinstance(state, str) and state == ''):
            state = np.zeros(self.state_dim, dtype=np.float32)
        state = np.array(state, dtype=np.float32)
        if len(state.shape) == 1:
            state = state[None, :]  # 增加 batch 维

        state_tensor = tf.convert_to_tensor(state, dtype=tf.float32)

        # --- 计算 policy & value ---
        value = agent.critic(state_tensor)
        value = tf.squeeze(value).numpy()  # 去掉多余维度

        # --- 用模型预测动作均值 ---
        mean = agent.actor(state_tensor)  # ✅ 模型输出 mean 向量
        log_std = tf.Variable(tf.zeros_like(mean), trainable=False)
        std = tf.exp(log_std)

        dist = tfp.distributions.Normal(mean, std)
        action = dist.sample()
        action_prob = tf.nn.softmax(action).numpy().flatten()
        #action_prob = tf.reduce_prod(dist.prob(action), axis=-1)

        action = tf.squeeze(action).numpy()
        action_prob = tf.squeeze(action_prob).numpy()

        # 防止空值
        #if np.any(np.isnan(action)):
        #    action = np.zeros(self.action_dim, dtype=np.float32)
        #if np.any(np.isnan(action_prob)):
        #    action_prob = 1.0

        return action, action_prob,value

    def select_action_eps_greedy(self, agent, epsilon=0.1):
        """
        eps-greedy 動作選擇
        """
        state = np.array([self.health, self.temp, self.humid, self.water, self.light, self.co2, self.ph])
        action, action_prob, value = agent.select_action(state)
        probs = tf.nn.softmax(action).numpy().flatten()
        # --- 轉成 numpy ---
        if isinstance(probs, tf.Tensor):
            probs = probs.numpy()
        probs = np.array(probs).astype(np.float32).squeeze()

        # --- 檢查合法性 ---
        if probs.ndim != 1:
            raise ValueError(f"[select_action_eps_greedy] probs 維度錯誤: {probs.shape}")
        if not np.isclose(np.sum(probs), 1.0, atol=1e-3):
            probs = np.exp(probs - np.max(probs))
            probs /= np.sum(probs)

        # --- 探索 vs 利用 ---
        if random.random() < epsilon:
            action_idx = random.randint(0, len(probs) - 1)
        else:
            action_idx = int(np.argmax(probs))

        # --- one-hot 動作 ---
        action_vec = np.zeros_like(probs)
        action_vec[action_idx] = 1.0

        return action_vec, action_prob, value

    def step(self, action ,  params=None, true_label=None):
        # Ensure params is not None
        if params is None:
            params = {
                "energy_penalty": 0.1,
                "switch_penalty_per_toggle": 0.2,
                "vpd_penalty": 2.0,
                "light_penalty": 0.5,
                "co2_penalty": 0.3,
                "flower_bonus": 0.5,
                "seed_bonus": 0.5,
                "grow_bonus": 0.5
            }

        if hasattr(self, 'prev_action'):
            # If prev_action is a numpy array, convert it to integer
            if isinstance(self.prev_action, np.ndarray):
                if self.prev_action.size == 1:
                    self.prev_action = int(self.prev_action.item())
                else:
                    self.prev_action = int(np.argmax(self.prev_action))

            # Get previous action tuple safely
            prev_action_tuple =self.prev_action # self.action_mapping.get(self.prev_action, (0, 0, 0, 0,0, 0, 0, 0))
        else:
            # Initialize prev_action if it doesn't exist
            self.prev_action = 0
            prev_action_tuple = (0, 0, 0, 0,0, 0, 0, 0)

        #heat , ac, humi, dehumi,waterpomb,light,carbon,ph =action # self.action_mapping[action]
        mode_param = self.mode_params[self.mode]
        temp_min, temp_max   = mode_param["temp_range"]
        humid_min, humid_max = mode_param["humid_range"]
        soil_min, soil_max   = mode_param["soil_range"]
        light_min, light_max = mode_param["light_range"]
        co2_min, co2_max     = mode_param["co2_range"]
        ph_min, ph_max       = mode_param["ph_range"]
        vpd_min, vpd_max     = mode_param["vpd_range"]
        # Calculate enhanced VPD
        vpd_current = self._calculate_enhanced_vpd(self.temp, self.humid, self.light, self.co2)



        optimal_conditions = 0
        if temp_min <= self.temp <= temp_max: optimal_conditions += 1
        if humid_min <= self.humid <= humid_max: optimal_conditions += 1
        if soil_min <= self.water <= soil_max: optimal_conditions += 1
        if light_min <= self.light <= light_max: optimal_conditions += 1
        if co2_min <= self.co2 <= co2_max: optimal_conditions += 1
        if ph_min <= self.ph <= ph_max: optimal_conditions += 1
        if vpd_min <= vpd_current <= vpd_max: optimal_conditions += 1



        #dev_heat, dev_ac, dev_humid, dev_dehumi, dev_waterpomb, dev_light, dev_carbon, dev_ph = action_vec
        action = np.array(action, dtype=float)
        self._apply_action(action)
        #current_state_tuple  = (self.health, self.temp, self.humid, self.water, self.light, self.co2, self.ph)
        current_action_tuple = (self.device_action["heat"],
                                self.device_action["ac"],
                                self.device_action["humid"],
                                self.device_action["dehumi"],
                                self.device_action["waterpomb"],
                                self.device_action["light"],
                                self.device_action["carbon"],
                                self.device_action["ph"]
                                )

        # Environment dynamics
        #self.temp += (-0.5 if dev_ac == 1 else 0.2) + (0.5 if dev_heat == 1 else 0.0)
        #self.humid += (0.05 if dev_humid == 1 else -0.02) + (-0.03 if dev_heat == 1 else 0.0) + (-0.05 if dev_dehumi == 1 else 0.0)
        # 模擬環境變化
        # --- 環境動態 ---
        inertia = 0.7
        effective_action = inertia * np.array(prev_action_tuple) + (1 - inertia) * np.array(current_action_tuple)

        # ---- Apply effective action ----
        self.temp += 0.5 * effective_action[0] - 0.4 * effective_action[1]
        self.humid += 0.3 * effective_action[2] - 0.3 * effective_action[3]
        self.water += 0.2 * effective_action[4] - 0.05
        self.light += 10 * effective_action[5] - 5
        self.co2 += 50 * effective_action[6] - 20
        self.ph += 0.1 * (effective_action[7] - 0.5)
        #self.temp += 0.5 * self.device_state["heat"] - 0.4 * self.device_state["ac"]
        #self.humid += 0.3 * self.device_state["humid"] - 0.3 * self.device_state["dehumi"]
        #self.water += 0.2 * self.device_state["waterpomb"] - 0.05
        #self.light += 10 * self.device_state["light"] - 5
        #self.co2 += 50 * self.device_state["carbon"] - 20
        #self.ph += 0.1 * (self.device_state["ph"] - 0.5)
        # Light and CO2 natural changes
        self.light += np.random.normal(0, 20)
        self.co2 += np.random.normal(0, 10)

        # Get ideal environment parameters based on current mode


        if optimal_conditions >= 4:
            self.health = 0
        elif optimal_conditions >= 2:
            self.health = 1
        else:
            self.health = 2


        # Calculate reward
        health_reward = {0: 2.0, 1: 0.5, 2: -1.0}[self.health]
        energy_cost = params.get("energy_penalty", 0.1) * np.sum(current_action_tuple)

        # Fixed switch penalty calculation
        switch_penalty = params.get("switch_penalty_per_toggle", 0.2) * np.sum(
            np.abs(np.array(current_action_tuple) - np.array(prev_action_tuple))
        )

        # Environmental factor rewards
        vpd_ideal   =  (vpd_min + vpd_max) / 2
        light_ideal =  (light_min + light_max) / 2
        co2_ideal   =  (co2_min + co2_max) / 2

        vpd_reward = -abs(vpd_current - vpd_ideal) * params.get("vpd_penalty", 2.0)
        light_reward = -abs(self.light - light_ideal) * params.get("light_penalty", 0.5)
        co2_reward = -abs(self.co2 - co2_ideal) * params.get("co2_penalty", 0.3)
        self.temp  =self.pid_map(self.temp, temp_min, temp_max, 0, 1)
        self.humid =self.pid_map(self.humid,humid_min,humid_max, 0, 1)
        self.water  =self.pid_map(self.water, soil_min, soil_max, 0, 1)
        self.light =self.pid_map(self.light,light_min,light_max, 0, 1)
        self.co2   =self.pid_map(self.co2,  co2_min,  co2_max, 0, 1) #np.clip(self.co2, 300, 1200)
        self.ph    =self.pid_map(self.ph,   ph_min,   ph_max, 0, 1)
        # Update sequence data
        new_data_point = np.array([self.health, self.temp, self.humid, self.water, self.light, self.co2, self.ph])
        self.update_sequence(new_data_point)

        # LLL model prediction
        seq_input_tf = tf.convert_to_tensor(self.current_sequence, dtype=tf.float32)

        if true_label is not None and not isinstance(true_label, tf.Tensor):
            true_label_tf = tf.convert_to_tensor(true_label, dtype=tf.int32)
        else:
            true_label_tf = true_label

        # Use update_lll_model method to get soft labels
        soft_label = self.update_lll_model(seq_input_tf, true_label_tf)
        flower_prob = soft_label[2]
        learning_reward = 0
        if true_label is not None:
            pred_class = np.argmax(soft_label)
            true_class = true_label if isinstance(true_label, (int, np.integer)) else true_label.numpy()
            learning_reward = 0.5 if pred_class == true_class else -0.3

        # Soft label reward
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

        # Store the action as integer
        self.prev_action = action
        self.t += 1
        #done = self.t >= self.seq_len
        done=self._check_done(new_data_point)
        info = {
            "health_status": self.health,
            "temp": self.temp,
            "humid": self.humid,
            "soil": self.water,
            "light": self.light,
            "co2": self.co2,
            "ph": self.ph,
            "vpd": vpd_current,
            "latent_soft_label": soft_label,
            "flower_prob": flower_prob,
            "learning_reward": learning_reward,
            "soft_label_bonus": soft_label_bonus,
            "health_status_text": ["健康", "亞健康", "不健康"][self.health],
            "optimal_conditions": optimal_conditions
        }
        #state = [health, temp, humid, light, co2, ph, water]
        #health, temp, humid, light, co2, ph, water   =self._get_state()
        #return [health, temp, humid, light, co2, ph, water], reward, done, info
        return self._get_state(), reward, done, info