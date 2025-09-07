import pandas as pd
from tensorflow import keras
from tensorflow.keras import optimizers

from sklearn.model_selection import train_test_split

from utils_QModel import QModel
# 依赖：NUM_SWITCH, DATA_DIR, SEQ_LEN, NUM_FEATURES 等
from utils_fisher import *                 # 如果无需本文件的函数，可以移除这行
from plant_analysis import *
# ============== 配置 ==============
ENCODER_LATENT_DIM = 16
Q_HIDDEN = [64, 64]

# 注意：NUM_ACTIONS 必须由 NUM_SWITCH 推导
NUM_ACTIONS = 2 ** NUM_SWITCH

LR_Q = 1e-3
SARSA_EPISODES = 200
BATCH_MAX = 32
GAMMA = 0.95
EPS_START = 0.3
EPS_END = 0.1
EPS_DECAY = 0.995
ACTION_COST = 0.05

# ============== 混合精度（此处保持 float32 更稳） ==============
from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy('float32')

# 数据文件列表
files = [os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) if f.endswith(".csv")]

def normalize_row(row):
    row = row.astype(np.float32)
    row[0] /= 40.0     # temp
    row[1] /= 100.0    # humid
    row[2] /= 1000.0   # light
    return row

def make_windows(X, seq_len, step_size=1):
    """
    X: (T, F) -> windows: (N, seq_len, F), idx: list of start indices
    N = floor((T - seq_len)/step_size) + 1
    """
    T = X.shape[0]
    if T < seq_len:
        return np.empty((0, seq_len, X.shape[1]), dtype=np.float32), []
    starts = list(range(0, T - seq_len + 1, step_size))
    windows = np.stack([X[s:s+seq_len] for s in starts], axis=0).astype(np.float32)
    return windows, starts


import numpy as np
import json
import os
from collections import defaultdict
#print(f"✅ QModel 初始化完成: 動作數={action_size}, 參數={params}")

import time
from datetime import datetime
import matplotlib.pyplot as plt

class QModel_aug(QModel):
    def __init__(self, state_size=3, action_size=16, learning_rate=0.1, gamma=0.9,
                 epsilon=0.3, config_file="best_params.json"):
        self.latent_dim=ENCODER_LATENT_DIM
        self.q_net = self.build_q_network(self.latent_dim, action_size)
        self.optimizer = optimizers.Adam(LR_Q)
        self.eps = EPS_START
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate  # 属性名是 learning_rate
        self.gamma = gamma  # 属性名是 gamma
        self.epsilon = epsilon
        self.q_table = np.zeros((state_size, action_size))  # 或者其他的Q函数表示

        """
        初始化 Q-learning 模型
        """
        self.config_file = config_file


        # 加載或設置參數
        self.best_params = self._load_best_params()
        params = self._get_training_params()

        self.learning_rate = params.get('learning_rate', learning_rate)
        self.gamma = params.get('gamma', gamma)
        self.epsilon = params.get('initial_epsilon', epsilon)
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995

        # 初始化 Q-table：每個狀態對應一個大小為 action_size 的數組
        self.q_table = defaultdict(lambda: np.zeros(action_size))

        print(f"✅ QModel 初始化完成: 動作數={action_size}, 參數={params}")




        self.exploration_stats = {
            'total_choices': 0,
            'exploration_choices': 0,
            'exploitation_choices': 0,
            'action_distribution': np.zeros(action_size),
            'last_choice_type': None
        }


        # 訓練過程監控
        self.training_history = {
            'episodes': [],
            'rewards': [],
            'avg_rewards': [],
            'epsilons': [],
            'q_table_sizes': [],
            'exploration_rates': [],
            'losses': [],
            'timestamps': [],
            'action_distributions': []
        }

        self.current_episode = 0
        self.episode_rewards = []
        self.episode_start_time = time.time()

        print(f"✅ QModel 初始化完成: 動作數={action_size}, 參數={params}")
        super().__init__()
        self.meta_model = keras.models.load_model("meta_model.h5")
        self.hvac_dense_layer = self.meta_model.get_layer("hvac_dense")  # lstm_encoder classifier
        self.encoder = keras.models.Model(inputs=self.meta_model.input, outputs=self.hvac_dense_layer.output)
    def start_episode(self):
        """開始新的訓練回合"""
        self.current_episode += 1
        self.episode_rewards = []
        self.episode_start_time = time.time()
        self.episode_exploration_count = 0
        self.episode_exploitation_count = 0

    def end_episode(self):
        """結束當前訓練回合並記錄統計"""
        if not self.episode_rewards:
            return

        total_reward = sum(self.episode_rewards)
        avg_reward = total_reward / len(self.episode_rewards)
        episode_duration = time.time() - self.episode_start_time

        # 記錄訓練歷史
        self.training_history['episodes'].append(self.current_episode)
        self.training_history['rewards'].append(total_reward)
        self.training_history['avg_rewards'].append(avg_reward)
        self.training_history['epsilons'].append(self.epsilon)
        self.training_history['q_table_sizes'].append(len(self.q_table))
        self.training_history['timestamps'].append(datetime.now())
        self.training_history['losses'].append(0)  # 可以根據需要計算損失

        # 計算探索率
        total_choices = self.episode_exploration_count + self.episode_exploitation_count
        if total_choices > 0:
            exploration_rate = (self.episode_exploration_count / total_choices) * 100
        else:
            exploration_rate = 0

        self.training_history['exploration_rates'].append(exploration_rate)

        # 記錄動作分佈
        action_dist = self.exploration_stats['action_distribution'].copy()
        self.training_history['action_distributions'].append(action_dist)

    def update(self, state, action, reward, next_state, next_action=None):
            """
            更新 Q-table 並記錄訓練數據
            """
            # 記錄獎勵
            self.episode_rewards.append(reward)

            # 記錄探索/利用
            discrete_state = self._discretize_state(state)
            q_values = self.q_table[discrete_state]
            if np.random.random() < self.epsilon:
                self.episode_exploration_count += 1
            else:
                self.episode_exploitation_count += 1

            # 原有的更新邏輯
            discrete_next_state = self._discretize_state(next_state)
            current_q = self.q_table[discrete_state][action]

            if next_action is not None:
                next_q = self.q_table[discrete_next_state][next_action]
            else:
                next_q = np.max(self.q_table[discrete_next_state])

            new_q = current_q + self.learning_rate * (
                    reward + self.gamma * next_q - current_q
            )

            self.q_table[discrete_state][action] = new_q

            # 衰減探索率
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

    def check_training_process(self, detailed=False, plot=False):
        """
        檢查訓練過程狀態

        Args:
            detailed: 是否顯示詳細信息
            plot: 是否生成訓練曲線圖

        Returns:
            訓練過程統計字典
        """
        if not self.training_history['episodes']:
            return {"status": "尚未開始訓練"}

        current_episode = self.current_episode
        latest_reward = self.training_history['rewards'][-1] if self.training_history['rewards'] else 0
        avg_reward = self.training_history['avg_rewards'][-1] if self.training_history['avg_rewards'] else 0

        stats = {
            'current_episode': current_episode,
            'total_episodes': len(self.training_history['episodes']),
            'latest_reward': round(latest_reward, 2),
            'latest_avg_reward': round(avg_reward, 2),
            'current_epsilon': round(self.epsilon, 4),
            'q_table_size': len(self.q_table),
            'training_duration': self._get_training_duration(),
            'overall_avg_reward': round(np.mean(self.training_history['rewards']), 2) if self.training_history[
                'rewards'] else 0,
            'best_reward': round(max(self.training_history['rewards']), 2) if self.training_history[
                'rewards'] else 0,
            'convergence_status': self._check_convergence()
        }

        if detailed:
            # 添加詳細統計
            stats.update({
                'reward_trend': self._get_reward_trend(),
                'exploration_trend': self._get_exploration_trend(),
                'recent_performance': self._get_recent_performance(10),
                'action_distribution': self._get_current_action_distribution()
            })

        if plot:
            self._plot_training_progress()

        return stats

    def _get_training_duration(self):
        """計算總訓練時間"""
        if not self.training_history['timestamps']:
            return "0s"

        start_time = self.training_history['timestamps'][0]
        end_time = self.training_history['timestamps'][-1]
        duration = (end_time - start_time).total_seconds()

        if duration < 60:
            return f"{int(duration)}s"
        elif duration < 3600:
            return f"{int(duration // 60)}m {int(duration % 60)}s"
        else:
            return f"{int(duration // 3600)}h {int((duration % 3600) // 60)}m"

    def _check_convergence(self):
        """檢查模型是否收斂"""
        if len(self.training_history['rewards']) < 20:
            return "需要更多訓練數據"

        recent_rewards = self.training_history['rewards'][-10:]
        reward_std = np.std(recent_rewards)

        if reward_std < 5:  # 獎勵波動很小
            return "可能已收斂"
        elif self.epsilon <= self.epsilon_min + 0.01:
            return "探索率已穩定"
        else:
            return "仍在學習中"

    def _get_reward_trend(self):
        """獲取獎勵趨勢"""
        if len(self.training_history['rewards']) < 5:
            return "數據不足"

        recent_rewards = self.training_history['rewards'][-5:]
        if all(recent_rewards[i] <= recent_rewards[i + 1] for i in range(len(recent_rewards) - 1)):
            return "上升"
        elif all(recent_rewards[i] >= recent_rewards[i + 1] for i in range(len(recent_rewards) - 1)):
            return "下降"
        else:
            return "波動"

    def _get_exploration_trend(self):
            """獲取探索趨勢"""
            if len(self.training_history['exploration_rates']) < 5:
                return "數據不足"

            recent_rates = self.training_history['exploration_rates'][-5:]
            return round(np.mean(recent_rates), 2)

    def _get_recent_performance(self, n=10):
        """獲取最近n個回合的性能"""
        if len(self.training_history['rewards']) < n:
            n = len(self.training_history['rewards'])

        recent_rewards = self.training_history['rewards'][-n:]
        return {
            'avg_reward': round(np.mean(recent_rewards), 2),
            'min_reward': round(min(recent_rewards), 2),
            'max_reward': round(max(recent_rewards), 2),
            'std_reward': round(np.std(recent_rewards), 2)
        }

    def _get_current_action_distribution(self):
        """獲取當前動作分佈"""
        total = sum(self.exploration_stats['action_distribution'])
        if total == 0:
            return {f'action_{i}': 0 for i in range(self.action_size)}

        return {
            f'action_{i}': round((count / total) * 100, 2)
            for i, count in enumerate(self.exploration_stats['action_distribution'])
        }

    def _plot_training_progress(self):
        """繪製訓練進度圖"""
        if len(self.training_history['episodes']) < 2:
            print("需要更多數據來繪圖")
            return

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

        # 獎勵曲線
        ax1.plot(self.training_history['episodes'], self.training_history['rewards'])
        ax1.set_title('Total Reward per Episode')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Reward')
        ax1.grid(True)

        # 平均獎勵曲線
        ax2.plot(self.training_history['episodes'], self.training_history['avg_rewards'])
        ax2.set_title('Average Reward per Episode')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Average Reward')
        ax2.grid(True)

        # 探索率曲線
        ax3.plot(self.training_history['episodes'], self.training_history['exploration_rates'])
        ax3.set_title('Exploration Rate')
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('Exploration Rate (%)')
        ax3.grid(True)

        # Q-table 大小曲線
        ax4.plot(self.training_history['episodes'], self.training_history['q_table_sizes'])
        ax4.set_title('Q-table Size')
        ax4.set_xlabel('Episode')
        ax4.set_ylabel('Number of States')
        ax4.grid(True)

        plt.tight_layout()
        plt.savefig('training_progress.png')
        plt.close()

        print("訓練進度圖已保存為 'training_progress.png'")

    def get_training_summary(self):
            """獲取訓練摘要報告"""
            stats = self.check_training_process(detailed=True)

            summary = f"""
            🎯 訓練過程摘要
            ========================
            總回合數: {stats['total_episodes']}
            當前回合: {stats['current_episode']}
            訓練時長: {stats['training_duration']}

            📊 獎勵統計:
            最新獎勵: {stats['latest_reward']}
            平均獎勵: {stats['overall_avg_reward']}
            最佳獎勵: {stats['best_reward']}
            獎勵趨勢: {stats['reward_trend']}

            ⚙️ 模型狀態:
            Q-table 大小: {stats['q_table_size']} 個狀態
            當前探索率: {stats['current_epsilon']}
            探索趨勢: {stats['exploration_trend']}%
            收斂狀態: {stats['convergence_status']}

            🎯 最近表現:
            平均獎勵: {stats['recent_performance']['avg_reward']}
            波動範圍: ±{stats['recent_performance']['std_reward']}
            """

            return summary




    def choose_action(self, state):
        """
        根據 epsilon-greedy 策略選擇動作
        """
        discrete_state = self._discretize_state(state)

        self.exploration_stats['total_choices'] += 1

        if np.random.random() < self.epsilon:
            # 探索：隨機選擇動作
            action = np.random.randint(self.action_size)
            self.exploration_stats['exploration_choices'] += 1
            self.exploration_stats['last_choice_type'] = 'exploration'
        else:
            # 利用：選擇Q值最大的動作
            q_values = self.q_table[discrete_state]
            action = np.argmax(q_values)
            self.exploration_stats['exploitation_choices'] += 1
            self.exploration_stats['last_choice_type'] = 'exploitation'

        # 記錄動作分佈
        self.exploration_stats['action_distribution'][action] += 1

        return action

    def check_exploration(self, detailed=False):
        """
        檢查探索狀態和統計信息

        Args:
            detailed: 是否顯示詳細信息

        Returns:
            探索狀態統計字典
        """
        total = self.exploration_stats['total_choices']
        if total == 0:
            return {"status": "尚未進行任何選擇"}

        exploration_rate = (self.exploration_stats['exploration_choices'] / total) * 100
        exploitation_rate = (self.exploration_stats['exploitation_choices'] / total) * 100

        stats = {
            'total_choices': total,
            'exploration_choices': self.exploration_stats['exploration_choices'],
            'exploitation_choices': self.exploration_stats['exploitation_choices'],
            'exploration_rate_percent': round(exploration_rate, 2),
            'exploitation_rate_percent': round(exploitation_rate, 2),
            'current_epsilon': round(self.epsilon, 4),
            'epsilon_min': self.epsilon_min,
            'last_choice_type': self.exploration_stats['last_choice_type'],
            'is_exploring': exploration_rate > 20  # 如果探索率大於20%，認為還在探索階段
        }

        if detailed:
            # 添加詳細的動作分佈信息
            action_distribution = self.exploration_stats['action_distribution']
            action_percentages = (action_distribution / total) * 100

            stats['action_distribution'] = {
                f'action_{i}': {
                    'count': int(action_distribution[i]),
                    'percentage': round(action_percentages[i], 2)
                } for i in range(self.action_size)
            }

            # 添加Q-table統計
            stats['q_table_size'] = len(self.q_table)
            stats['average_q_values'] = self._get_average_q_values()

        return stats

    def _get_average_q_values(self):
        """計算所有狀態的平均Q值"""
        if not self.q_table:
            return {f'action_{i}': 0 for i in range(self.action_size)}

        total_q = np.zeros(self.action_size)
        count = 0

        for state_q in self.q_table.values():
            total_q += state_q
            count += 1

        return {f'action_{i}': round(total_q[i] / count, 4) for i in range(self.action_size)}

    def reset_exploration_stats(self):
        """重置探索統計"""
        self.exploration_stats = {
            'total_choices': 0,
            'exploration_choices': 0,
            'exploitation_choices': 0,
            'action_distribution': np.zeros(self.action_size),
            'last_choice_type': None
        }
        print("探索統計已重置")

    def set_epsilon(self, new_epsilon):
        """手動設置epsilon值"""
        old_epsilon = self.epsilon
        self.epsilon = max(self.epsilon_min, min(new_epsilon, 1.0))
        print(f"Epsilon 從 {old_epsilon} 調整為 {self.epsilon}")

    def get_exploration_recommendation(self):
            """
            根據當前狀態給出探索建議
            """
            stats = self.check_exploration()

            if stats['total_choices'] < 50:
                return "訓練初期，建議保持較高探索率"

            if stats['exploration_rate_percent'] < 5:
                return "探索率過低，建議適當增加探索"
            elif stats['exploration_rate_percent'] > 40:
                return "探索率過高，建議減少探索，增加利用"
            else:
                return "探索率在合理範圍內"




    def _load_best_params(self):
        """從文件加載最佳參數"""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    params = json.load(f)
                print(f"✅ 已加載最佳參數: {params}")
                return params
            return None
        except Exception as e:
            print(f"❌ 加載參數失敗: {e}")
            return None

    def _get_training_params(self):
        """獲取訓練參數"""
        if self.best_params:
            return self.best_params
        else:
            return {
                'learning_rate': 0.1,
                'gamma': 0.9,
                'initial_epsilon': 0.3
            }

    def save_best_params(self, params):
        """保存最佳參數到文件"""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(params, f, indent=4)
            print(f"✅ 最佳參數已保存到 {self.config_file}")
            self.best_params = params
        except Exception as e:
            print(f"❌ 保存參數失敗: {e}")

    def _discretize_state0(self, state):
        """
        離散化狀態
        """
        if isinstance(state, (list, tuple, np.ndarray)):
            if len(state) == 0:
                return (0,)

            # 處理多維狀態
            if isinstance(state[0], (list, tuple, np.ndarray)):
                # 如果是二維數組，展平並離散化
                flattened = np.array(state).flatten()
                discrete_tuple = tuple(int(np.clip(x * 100, 0, 99)) for x in flattened[:3])  # 限制特徵數量
            else:
                # 一維數組
                discrete_tuple = tuple(int(np.clip(x * 100, 0, 99)) for x in state[:3])  # 限制特徵數量
            return discrete_tuple
        else:
            # 單一數值
            return (int(np.clip(state * 100, 0, 99)),)



    def get_q_values(self, state):
        """獲取指定狀態的Q值"""
        discrete_state = self._discretize_state(state)
        return self.q_table[discrete_state].copy()

    def get_policy(self, state):
        """獲取指定狀態的策略（動作概率分佈）"""
        q_values = self.get_q_values(state)
        # 使用softmax將Q值轉換為概率分佈
        exp_q = np.exp(q_values - np.max(q_values))  # 數值穩定性
        return exp_q / np.sum(exp_q)

    def reset_epsilon(self, new_epsilon=None):
        """重置探索率"""
        if new_epsilon is not None:
            self.epsilon = new_epsilon
        else:
            params = self._get_training_params()
            self.epsilon = params.get('initial_epsilon', 0.3)

    def get_stats(self):
        """獲取模型統計信息"""
        return {
            'q_table_size': len(self.q_table),
            'epsilon': self.epsilon,
            'learning_rate': self.learning_rate,
            'gamma': self.gamma,
            'action_size': self.action_size
        }

    def debug_state(self, state):
        """調試狀態離散化"""
        discrete_state = self._discretize_state(state)
        q_values = self.q_table[discrete_state]
        return {
            'original_state': state,
            'discrete_state': discrete_state,
            'q_values': q_values,
            'recommended_action': np.argmax(q_values)
        }


    def _discretize_state(self, state):
        discrete = 0
        if isinstance(state, (list, tuple, np.ndarray)):
            # 如果狀態是數組，處理每個元素
            for i, value in enumerate(state):
                if isinstance(value, (list, tuple, np.ndarray)):
                    # 如果元素本身也是數組，遞歸處理或特殊處理
                    discrete += int(np.mean(value) * 100) * (10 ** (i * 2))
                else:
                    # 單一數值
                    discrete += int(value * 100) * (10 ** (i * 2))
        else:
            # 單一數值狀態
            discrete = int(state * 100)

        return discrete


    # ---------- Q网络 ----------
    def build_q_network(self, latent_dim, num_actions):
        inp = layers.Input(shape=(latent_dim,), name="latent_in")
        x = inp
        for h in Q_HIDDEN:
            x = layers.Dense(h, activation="relu")(x)
        out = layers.Dense(num_actions, activation=None, name="q_out")(x)
        return models.Model(inp, out, name="q_net")

    # ---------- 工具 ----------
    @staticmethod
    def action_int_to_bits(a_int):
        return np.array([(a_int >> i) & 1 for i in range(NUM_SWITCH)], dtype=np.float32)

    @staticmethod
    def select_action_batch(q_values, eps):
        # q_values: (B, A)
        if q_values.shape[0] == 0:
            return np.zeros((0,), dtype=np.int32)
        greedy = np.argmax(q_values, axis=1).astype(np.int32)
        rand_mask = (np.random.rand(q_values.shape[0]) < eps)
        random_actions = np.random.randint(q_values.shape[1], size=q_values.shape[0])
        actions = np.where(rand_mask, random_actions, greedy).astype(np.int32)
        return actions

    @staticmethod
    def compute_reward_batch(labels, actions_bits):
        """
        labels: (B,) 值域示例 {0,1,其他}
        actions_bits: (B, NUM_SWITCH)
        """
        # 示例奖励：0 -> +1, 1 -> -1, else -> -2
        rewards = np.where(labels == 0, 1.0, np.where(labels == 1, -1.0, -2.0))
        rewards = rewards - ACTION_COST * np.sum(actions_bits, axis=1)
        return rewards.astype(np.float32)

    def encode_sequence(self, encoder, X_seq):
        """
        X_seq: (B, seq_len, F) 或 (1, seq_len, F)
        返回 (B, latent_dim)
        """
        return encoder(X_seq).numpy().astype(np.float32)

    def encode_per_timestep_aug(self, encoder, raw, seq_len=None, step_size=1):
        """
        Args:
            encoder: 模型 (輸入 [B, seq_len, F] -> 輸出 latent)
            raw: numpy array, shape [T, F]
            seq_len: 每個窗口長度 (int 或 None)，None 時自動決定
            step_size: 滑動步長

        Returns:
            s_latent_seq: shape [N, D]，所有窗口 latent
            starts: 每個窗口對應的起始索引
        """
        T = raw.shape[0]

        # --- 動態決定窗口長度 ---
        if seq_len is None:
            # 例如取 T 的 1/10，至少 5
            seq_len = max(10, T // 20)

        # --- 滑動窗口切片 ---
        starts = list(range(0, T - seq_len + 1, step_size))
        seqs = [raw[i:i + seq_len] for i in starts]

        if not seqs:
            return np.zeros((0, encoder.output_shape[-1])), []

        seqs = np.array(seqs)  # [N, seq_len, F]

        # --- 編碼 ---
        s_latent_seq = encoder.predict(seqs, verbose=0)  # [N, D]

        return s_latent_seq, starts


    @tf.function
    def _sarsa_batch_update_tf(self, s_latent, actions, s_next_latent, rewards):
        # s_latent: (B, D), actions: (B,), s_next_latent: (B, D), rewards: (B,)
        with tf.GradientTape() as tape:
            q_pred_all = self.q_net(s_latent, training=True)                 # (B, A)
            batch_idx = tf.range(tf.shape(actions)[0], dtype=tf.int32)
            q_pred = tf.gather_nd(q_pred_all, tf.stack([batch_idx, actions], axis=1))  # (B,)

            # target 使用 next state's max-Q（SARSA可改：用 next action 的 Q；此处做 Double-check）
            q_next_all = self.q_net(s_next_latent, training=False)           # (B, A)
            next_actions = tf.argmax(q_next_all, axis=1, output_type=tf.int32)
            q_next = tf.gather_nd(q_next_all, tf.stack([batch_idx, next_actions], axis=1))  # (B,)

            rewards = tf.cast(rewards, q_next.dtype)
            target = rewards + GAMMA * q_next                                # (B,)

            loss = tf.reduce_mean(tf.square(target - q_pred))
        grads = tape.gradient(loss, self.q_net.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.q_net.trainable_variables))
        return loss

    def sarsa_full_batch_robust(self, encoder, seq_len, step_size=1):
        """
        每次从 CSV 采样一段，构造 (s_t, a_t, r_{t+1}, s_{t+1}) 的全批量样本并更新。
        """
        for ep in range(1, SARSA_EPISODES + 1):
            df = pd.read_csv(np.random.choice(files))

            raw = df[["temp", "humid", "light", "ac", "heater", "dehum", "hum"]].values.astype(np.float32)
            labels = df["label"].values.astype(np.int32)

            # 归一化非控制量
            raw[:, 0] /= 40.0
            raw[:, 1] /= 100.0
            raw[:, 2] /= 1000.0

            # 计算所有窗口 latent
            s_latent_seq, starts = self.encode_per_timestep(encoder, raw)  # (N, D)
            N = s_latent_seq.shape[0]
            if N <= 1:
                # 不足以形成 (s_t, s_{t+1})
                continue

            # 预测当前 Q(s_t, ·)
            q_pred = self.q_net.predict(s_latent_seq[:-1], verbose=0)  # (N-1, A)
            # 探索噪声（很小）
            q_pred = q_pred + np.random.randn(*q_pred.shape).astype(np.float32) * 0.01

            # ε-greedy 选行动
            actions = self.select_action_batch(q_pred, self.eps)              # (N-1,)
            actions_bits = np.stack([self.action_int_to_bits(a) for a in actions], axis=0)  # (N-1, NUM_SWITCH)

            # 奖励：使用下一步窗口的标签（与 s_{t+1} 对齐）
            label_next_idx = [min(s + seq_len, len(labels) - 1) for s in starts[:-1]]
            rewards = self.compute_reward_batch(labels=np.array(label_next_idx, dtype=np.int32)*0 + labels[label_next_idx],
                                                actions_bits=actions_bits)

            # 构造 s_next
            s_next_latent_seq = s_latent_seq[1:]                              # (N-1, D)

            # 分批更新，避免过大 batch
            total_loss = 0.0
            B = BATCH_MAX
            for i in range(0, len(actions), B):
                j = min(i + B, len(actions))
                if j - i <= 0: continue
                loss = self._sarsa_batch_update_tf(
                    tf.convert_to_tensor(s_latent_seq[i:j], dtype=tf.float32),
                    tf.convert_to_tensor(actions[i:j], dtype=tf.int32),
                    tf.convert_to_tensor(s_next_latent_seq[i:j], dtype=tf.float32),
                    tf.convert_to_tensor(rewards[i:j], dtype=tf.float32),
                )
                total_loss += float(loss.numpy())

            # 衰减 epsilon
            self.eps = max(EPS_END, self.eps * EPS_DECAY)

            if ep % 10 == 0 or ep == 1:
                print(f"Episode {ep:03d}/{SARSA_EPISODES} | eps={self.eps:.3f} | loss≈{total_loss:.4f} | samples={len(actions)}")

    def rollout_meta_sarsa(self, X_train, X_val, encoder, seq_len, steps=30, step_size=1, epsilon=0.1):
        """
        仅用于快速 sanity check：基于编码的 ε-贪婪 rollout。
        """
        latents, starts = self.encode_per_timestep(encoder, X_train, seq_len=seq_len, step_size=step_size)
        if latents.shape[0] == 0:
            print("No latent windows; check seq_len/step_size.")
            return

        rewards_list, actions_list = [], []
        for t in range(min(steps, latents.shape[0])):
            q_vals = self.q_net.predict(latents[t:t+1], verbose=0)[0]
            q_vals = q_vals + np.random.randn(NUM_ACTIONS).astype(np.float32) * 0.01
            if np.random.rand() < epsilon:
                a = np.random.randint(NUM_ACTIONS)
            else:
                a = int(np.argmax(q_vals))
            bits = self.action_int_to_bits(a)
            actions_list.append(bits)

            if X_val is not None and len(X_val) > 0:
                idx = min(starts[t] + seq_len, len(X_val) - 1)
                r = self.compute_reward_batch(np.array([int(X_val[idx, 0]*0)]),  # 这里没有真实标签，仅演示；建议换成 df 的 label
                                              np.array([bits]))[0]
            else:
                r = 0.0
            rewards_list.append(r)

        print("Actions:", [a.tolist() for a in actions_list])
        print("Rewards:", rewards_list)
        # 创建DataFrame
        df = pd.DataFrame({
            'Action_0': [action[0] for action in actions_list],
            'Action_1': [action[1] for action in actions_list],
            'Action_2': [action[2] for action in actions_list],
            'Action_3': [action[3] for action in actions_list],
            'Reward': rewards_list
        })

        # 保存到CSV文件
        df.to_csv('actions_rewards.csv', index=False)
        print("数据已保存到 actions_rewards.csv")


# ============== 测试 rollout ==============
def test_rollout(qmodel, encoder, steps=30, save_path="rollout_results.csv"):
    df = pd.read_csv(np.random.choice(files))

    # 获取前10个时间步的数据作为初始状态
    initial_data = []
    for i in range(10):
        row = df.iloc[i][["temp", "humid", "light", "ac", "heater", "dehum", "hum"]].values.astype(np.float32)
        initial_data.append(normalize_row(row))

    # 创建滑动窗口缓冲区
    state_buffer = np.array(initial_data)  # (10, 7)

    print("\nTest rollout:")

    # 创建结果列表
    results = []

    for t in range(steps):
        # 使用最近10个时间步的数据作为输入
        s_latent = encoder(state_buffer[np.newaxis, :, :]).numpy()[0]  # (latent_dim,)

        qv = qmodel.q_net.predict(s_latent.reshape(1, -1), verbose=0)[0]
        a = int(np.argmax(qv))
        bits = qmodel.action_int_to_bits(a)

        # 获取下一个时间步的数据
        next_idx = min(t + 10, len(df) - 1)
        next_row = df.iloc[next_idx][["temp", "humid", "light", "ac", "heater", "dehum", "hum"]].values.astype(
            np.float32)
        next_row = normalize_row(next_row)
        next_row[3:3 + NUM_SWITCH] = bits  # 用动作替换开关状态

        # 更新状态缓冲区：移除最旧的数据，添加最新的数据
        state_buffer = np.vstack([state_buffer[1:], next_row[np.newaxis, :]])

        label_next = int(df.iloc[next_idx]["label"]) if "label" in df.columns else 0
        r = qmodel.compute_reward_batch(np.array([label_next]), np.array([bits]))[0]

        # 打印结果
        print(f"t={t:02d} action={a:02d} bits={bits.tolist()} reward={r:.3f} label_next={label_next}")

        # 保存结果到列表
        results.append({
            'time_step': t,
            'action': a,
            'ac': bits[0],
            'heater': bits[1],
            'dehum': bits[2],
            'hum': bits[3],
            'reward': r,
            'label_next': label_next,
            'q_value_max': np.max(qv),
            'temp': next_row[0],
            'humid': next_row[1],
            'light': next_row[2]
        })

    # 将结果转换为DataFrame并保存为CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv(save_path, index=False)
    print(f"\n结果已保存到: {save_path}")

    return results_df


def test_rollout_detailed(qmodel, encoder, steps=30, save_path="rollout_detailed_results.csv"):
    df = pd.read_csv(np.random.choice(files))

    # 获取前10个时间步的数据作为初始状态
    initial_data = []
    for i in range(10):
        row = df.iloc[i][["temp", "humid", "light", "ac", "heater", "dehum", "hum"]].values.astype(np.float32)
        initial_data.append(normalize_row(row))

    state_buffer = np.array(initial_data)

    print("\nTest rollout (详细版):")

    results = []

    for t in range(steps):
        s_latent = encoder(state_buffer[np.newaxis, :, :]).numpy()[0]
        qv = qmodel.q_net.predict(s_latent.reshape(1, -1), verbose=0)[0]
        a = int(np.argmax(qv))
        bits = qmodel.action_int_to_bits(a)

        next_idx = min(t + 10, len(df) - 1)
        next_row = df.iloc[next_idx][["temp", "humid", "light", "ac", "heater", "dehum", "hum"]].values.astype(
            np.float32)
        original_next_row = next_row.copy()  # 保存原始值
        next_row = normalize_row(next_row)
        next_row[3:3 + NUM_SWITCH] = bits

        state_buffer = np.vstack([state_buffer[1:], next_row[np.newaxis, :]])

        label_next = int(df.iloc[next_idx]["label"]) if "label" in df.columns else 0
        r = qmodel.compute_reward_batch(np.array([label_next]), np.array([bits]))[0]

        print(f"t={t:02d} action={a:02d} bits={bits.tolist()} reward={r:.3f} label_next={label_next}")

        # 保存更详细的信息
        results.append({
            'time_step': t,
            'action': a,
            'action_binary': f"{a:04b}",
            'ac': bits[0],
            'heater': bits[1],
            'dehum': bits[2],
            'hum': bits[3],
            'reward': r,
            'label_next': label_next,
            'q_value_max': np.max(qv),
            'q_values': qv.tolist(),  # 所有Q值
            'temp': original_next_row[0],  # 原始温度值
            'humid': original_next_row[1],  # 原始湿度值
            'light': original_next_row[2],  # 原始光照值
            'temp_normalized': next_row[0],  # 标准化温度
            'humid_normalized': next_row[1],  # 标准化湿度
            'light_normalized': next_row[2]  # 标准化光照
        })

    # 保存为CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv(save_path, index=False, float_format='%.4f')
    print(f"\n详细结果已保存到: {save_path}")

    # 打印统计信息
    print(f"\n统计信息:")
    print(f"平均奖励: {results_df['reward'].mean():.3f}")
    print(f"总奖励: {results_df['reward'].sum():.3f}")
    print(f"动作分布:")
    print(results_df['action'].value_counts().sort_index())

    return results_df
# 使用示例
# results = test_rollout(qmodel, encoder, steps=30, save_path="rollout_results.csv")
# ============== 数据工具 ==============
def load_all_csvs(data_dir):
    dfs = [pd.read_csv(os.path.join(data_dir, f)) for f in os.listdir(data_dir) if f.endswith(".csv")]
    return pd.concat(dfs, axis=0).reset_index(drop=True)

def save_sarsa_tflite(q_net):
    converter = tf.lite.TFLiteConverter.from_keras_model(q_net)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    with open('../hvac_controller.tflite', 'wb') as f:
        f.write(tflite_model)
    q_net.save("hvac_controller.h5")
    print(f"Model size: {len(tflite_model)} bytes")


# ============== 主程序 ==============
def main():
    qmodel = QModel_aug(action_size=NUM_ACTIONS)
    # 训练
    #qmodel.sarsa_full_batch_robust(encoder, seq_len=SEQ_LEN, step_size=1)
    #save_sarsa_tflite(qmodel.q_net)

    # 基本版本
    #results = test_rollout(qmodel, encoder, steps=30, save_path="my_rollout_results.csv")

    # 详细版本
    #detailed_results = test_rollout_detailed(qmodel, encoder, steps=30, save_path="detailed_rollout_results.csv")
    # 或者单独运行某些调试功能
    #debugger = RL_Debugger()
    from fed_server.meta_sarsa.util_train_plant_env import PlantGrowthEnv
    env=PlantGrowthEnv()
    hyperparameter_tuning(qmodel,   env )
    #run_comprehensive_debug(qmodel, qmodel.encoder, env, files)
    # 使用示例
    #files = ["your_data_file1.csv", "your_data_file2.csv"]  # 你的文件列表
    #debugger = DataDrivenDebugger(files)
    #features, labels = debugger.load_and_preprocess_data()
if __name__ == "__main__":
    main()

