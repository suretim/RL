import numpy as np
import json
import os
from tensorflow import keras
from tensorflow.keras import optimizers

from sklearn.model_selection import train_test_split

# 依赖：NUM_SWITCH, DATA_DIR, SEQ_LEN, NUM_FEATURES 等
from utils_fisher import *                 # 如果无需本文件的函数，可以移除这行
from plant_analysis import *
from Q_model import QModel


class Plant_Health_Analysis:
    def __init__(self, config_file="best_params.json"):
        self.config_file = config_file
        self.best_params = self.load_best_params()

        # 初始化Q表和其他組件
        self.q_table = {}
        self._setup_algorithm()

    def _setup_algorithm(self):
        """設置算法參數"""
        params = self.get_training_params()

        self.learning_rate = params['learning_rate']
        self.gamma = params['gamma']
        self.epsilon = params['initial_epsilon']
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995

        print(f"算法參數設置完成: {params}")

    def save_best_params(self, params):
        """保存最佳參數到文件"""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(params, f, indent=4)
            print(f"✅ 最佳參數已保存到 {self.config_file}")
            self.best_params = params
            # 更新當前參數
            self._setup_algorithm()
        except Exception as e:
            print(f"❌ 保存參數失敗: {e}")

    def load_best_params(self):
        """從文件加載最佳參數"""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    params = json.load(f)
                print(f"✅ 已加載最佳參數: {params}")
                return params
            else:
                print("ℹ️  找不到參數文件，將使用默認參數")
                return None
        except Exception as e:
            print(f"❌ 加載參數失敗: {e}")
            return None

    def get_training_params(self):
        """獲取訓練參數"""
        if self.best_params:
            return self.best_params
        else:
            # 默認參數
            return {
                'learning_rate': 0.1,
                'gamma': 0.9,
                'initial_epsilon': 0.3
            }

    def _discretize_state(self, state):
        """離散化狀態（使用之前討論的方案4）"""
        if isinstance(state, (list, tuple, np.ndarray)):
            # 對數組中的每個元素進行離散化
            discrete_tuple = tuple(int(np.clip(x * 100, 0, 99)) for x in state)
            return discrete_tuple
        else:
            # 單一數值
            return (int(np.clip(state * 100, 0, 99)),)

    # 其他您原有的方法...
    def train(self, state, action, reward, next_state, next_action):

        meta_model = keras.models.load_model("../sarsa/meta_model.h5")
        meta_model.summary()
        hvac_dense_layer = meta_model.get_layer("hvac_dense")
        # inputs = tf.keras.Input(shape=(None, 7))  # None = 任意長度序列
        # x = tf.keras.layers.LSTM(64)(inputs)  # LSTM 可以處理可變長度
        # encoder = tf.keras.Model(inputs, x)

        encoder = keras.models.Model(inputs=meta_model.input, outputs=hvac_dense_layer.output)
        #(self, state_size=None, action_size=NUM_ACTIONS, learning_rate=0.01, gamma=0.99, epsilon=0.1, latent_dim=ENCODER_LATENT_DIM):

        ENCODER_LATENT_DIM = 16
        NUM_ACTIONS = 2 ** NUM_SWITCH

        qmodel = QModel(state_size=3, action_size=NUM_ACTIONS, learning_rate=0.01, gamma=0.99, epsilon=0.1)
        # 训练
        qmodel.sarsa_full_batch_robust(encoder, seq_len=SEQ_LEN, step_size=1)
        #save_sarsa_tflite(qmodel.q_net)
        from util_plant_env import PlantGrowthEnv
        env = PlantGrowthEnv()
        hyperparameter_tuning(qmodel,encoder, env)
        """訓練方法"""
        discrete_state = self._discretize_state(state)
        discrete_next_state = self._discretize_state(next_state)

        # SARSA更新規則
        current_q = self.q_table.get((discrete_state, action), 0)
        next_q = self.q_table.get((discrete_next_state, next_action), 0)

        new_q = current_q + self.learning_rate * (
                reward + self.gamma * next_q - current_q
        )

        self.q_table[(discrete_state, action)] = new_q

        # 衰減epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


# 使用示例
if __name__ == "__main__":
    # 初始化
    plant_analysis = Plant_Health_Analysis()

    # 使用最佳參數進行訓練
    state = [0.5, 0.3, 0.8]
    action = 1
    reward = 10
    next_state = [0.6, 0.2, 0.9]
    next_action = 2

    plant_analysis.train(state, action, reward, next_state, next_action)

    # 如果需要保存新的最佳參數
    new_best_params = {'learning_rate': 0.1, 'gamma': 0.9, 'initial_epsilon': 0.3}
    plant_analysis.save_best_params(new_best_params)