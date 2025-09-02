import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime

import tensorflow as tf
from tensorflow.keras import layers, Model

from util_check_estimator import *
from sklearn.preprocessing import StandardScaler
from IPython.display import clear_output
import time


class DataDrivenDebugger:
    def __init__(self, files):
        self.files = files
        self.history = {
            'rewards': [], 'q_values': [], 'actions': [],
            'losses': [], 'epsilons': [], 'correlations': [],
            'states': [], 'predictions': []
        }
        self.scaler = StandardScaler()

    def load_and_preprocess_data(self, file_idx=0):
        """加载和预处理数据"""
        df = pd.read_csv(self.files[file_idx])

        # 提取特征和标签
        features = df[["temp", "humid", "light", "ac", "heater", "dehum", "hum"]].values
        labels = df["label"].values if "label" in df.columns else np.zeros(len(df))

        # 标准化连续特征（前3个）
        features[:, :3] = self.scaler.fit_transform(features[:, :3])

        return features, labels

    def simulate_environment_step(self, current_idx, features,labels, action_bits):
        """模拟环境步骤 - 从数据中获取下一个状态"""
        next_idx = min(current_idx + 1, len(features) - 1)
        next_state = features[next_idx].copy()

        # 用动作替换开关状态（后4个特征）
        next_state[3:7] = action_bits

        # 获取奖励（如果有标签）
        reward = self.compute_reward(labels[next_idx] if 'labels' in locals() else 0, action_bits)

        done = (next_idx == len(features) - 1)

        return next_state, reward, done, next_idx

    def compute_reward(self, label, action_bits):
        """计算奖励（根据你的奖励函数）"""
        # 这里是示例奖励函数，你需要替换为你的实际实现
        base_reward = 1.0 if label == 0 else 0.1  # 无异常时高奖励

        # 惩罚过多的设备开启
        device_penalty = -0.1 * sum(action_bits)

        return base_reward + device_penalty




def advanced_diagnostics(qmodel, encoder, rollout_data):
    """
    高级诊断工具
    """
    print("\n" + "=" * 50)
    print("🔬 高级诊断分析")
    print("=" * 50)

    # 1. 分析Bellman误差
    analyze_bellman_error(qmodel, rollout_data)

    # 2. 检查值函数估计
    check_value_estimation(qmodel, encoder)

    # 3. 分析探索-利用平衡
    #analyze_exploration_exploitation(qmodel)

    # 4. 梯度分析
    analyze_gradients(qmodel)


def analyze_bellman_error(qmodel, data):
    """分析Bellman误差"""
    print("\n📉 Bellman误差分析:")

    # 计算TD误差
    td_errors = []
    for i in range(len(data) - 1):
        state = data.iloc[i][['temp', 'humid', 'light', 'ac', 'heater', 'dehum', 'hum']].values
        next_state = data.iloc[i + 1][['temp', 'humid', 'light', 'ac', 'heater', 'dehum', 'hum']].values
        reward = data.iloc[i + 1]['reward'] if 'reward' in data.columns else 0

        # 这里需要根据你的具体实现计算TD误差
        # td_error = ...
        # td_errors.append(td_error)

    if td_errors:
        print(f"  TD误差均值: {np.mean(td_errors):.4f}")
        print(f"  TD误差标准差: {np.std(td_errors):.4f}")


def analyze_gradients(qmodel):
    """分析梯度信息"""
    print("\n📊 梯度分析:")
    # 这里需要访问模型的梯度信息
    # 实际实现取决于你使用的深度学习框架

    print("  (需要根据具体框架实现梯度监控)")


# 5. 实时训练监控
def create_training_monitor():
    """创建实时训练监控"""

    class TrainingMonitor:
        def __init__(self):
            self.episode_rewards = []
            self.losses = []
            self.q_values = []

        def update(self, episode_reward, loss, avg_q):
            self.episode_rewards.append(episode_reward)
            self.losses.append(loss)
            self.q_values.append(avg_q)

            if len(self.episode_rewards) % 10 == 0:
                self.plot_progress()

        def plot_progress(self):
            clear_output(wait=True)
            fig, axes = plt.subplots(1, 3, figsize=(15, 4))

            axes[0].plot(self.episode_rewards)
            axes[0].set_title('Episode Rewards')
            axes[0].set_ylabel('Reward')

            axes[1].plot(self.losses)
            axes[1].set_title('Training Loss')
            axes[1].set_ylabel('Loss')
            axes[1].set_yscale('log')

            axes[2].plot(self.q_values)
            axes[2].set_title('Average Q-values')
            axes[2].set_ylabel('Q-value')

            plt.tight_layout()
            plt.show()

    return TrainingMonitor()

class SarsaAgent:
    def __init__(self, state_size, action_size, learning_rate=0.01, gamma=0.99, epsilon=0.1):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate # 属性名是 learning_rate
        self.gamma = gamma                 # 属性名是 gamma
        self.epsilon = epsilon
        self.q_table = np.zeros((state_size, action_size)) # 或者其他的Q函数表示

    def choose_action(self, state):
        """
        對於表格方法，需要將連續狀態離散化
        """
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_size)  # 探索

        else:
            # 將連續狀態離散化為整數索引
            discrete_state = self._discretize_state(state)
            return np.argmax(self.q_table[discrete_state, :])  # 利用

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


    def learn(self, state, action, reward, next_state, next_action, done):
        # Sarsa 更新公式
        current_q = self.q_table[state, action]
        if done:
            target = reward
        else:
            # 注意这里是 Q(S', A')，是下一个状态和“实际采取”的下一个动作
            target = reward + self.gamma * self.q_table[next_state, next_action]
        # 更新 Q值
        self.q_table[state, action] += self.learning_rate * (target - current_q)




def hyperparameter_tuning(qmodel, env ):
    """
    更高效的Sarsa超参数调试
    """
    # 減少參數組合，先測試最重要的參數
    hyperparams_to_test = {
        'learning_rate': [0.1, 0.01],  # 先測試兩個極端值
        'gamma': [0.9, 0.99],
        'initial_epsilon': [0.3, 0.1],
    }

    best_params = None
    best_performance = -float('inf')

    print("开始Sarsa超参数调试...")

    for lr in hyperparams_to_test['learning_rate']:
        for gamma in hyperparams_to_test['gamma']:
            for initial_eps in hyperparams_to_test['initial_epsilon']:

                print(f"測試 LR: {lr}, Gamma: {gamma}, ε: {initial_eps}")

                temp_model = qmodel  #SarsaAgent(state_size=latent_dim, action_size=16)
                temp_model.learning_rate = lr
                temp_model.gamma = gamma
                temp_model.epsilon = initial_eps
                # 使用固定衰減值
                temp_model.epsilon_decay = 0.995
                temp_model.epsilon_min = 0.01

                performance = test_hyperparameter_set(temp_model, qmodel.encoder, env, episodes=15)

                if performance > best_performance:
                    best_performance = performance
                    best_params = {
                        'learning_rate': lr,
                        'gamma': gamma,
                        'initial_epsilon': initial_eps
                    }
                    print(f"↑ 新的最佳! 性能: {performance:.3f}")

    print(f"\n第一階段最佳参数: {best_params}")

    # 可以在找到大致範圍後，進行第二輪更精細的搜索
    return best_params


def test_hyperparameter_set(model, encoder, env, episodes=10):
    """测试一组超参数的性能"""
    total_reward = 0
    window_size = encoder.input_shape[1]
    feature_keys = ['temp', 'humid', 'light', 'ac', 'heater', 'dehum', 'hum']

    for episode in range(episodes):
        state = env.reset()
        episode_reward = 0
        state_buffer = []

        # 預填充緩衝區
        for _ in range(window_size):
            # 檢查 state 是否包含所有需要的鍵
            if not all(key in state for key in feature_keys):
                missing = [key for key in feature_keys if key not in state]
                print(f"警告: state 缺少鍵: {missing}")
                # 使用默認值填充
                state_array = np.zeros(len(feature_keys))
            else:
                state_array = np.array([state[key] for key in feature_keys])

            state_buffer.append(state_array)
            next_state, reward, done, _ = env.step(0)
            state = next_state
            if done:
                break

        for step in range(100):
            state_sequence = np.array(state_buffer).reshape(1, window_size, -1)
            state_latent = encoder(state_sequence).numpy()

            action = model.choose_action(state_latent)
            next_state, reward, done, _ = env.step(action)

            # 檢查 next_state 是否包含所有需要的鍵
            if not all(key in next_state for key in feature_keys):
                missing = [key for key in feature_keys if key not in next_state]
                print(f"錯誤: next_state 缺少鍵: {missing}")
                # 使用緩衝區最後一個狀態或零填充
                next_state_array = state_buffer[-1] if state_buffer else np.zeros(len(feature_keys))
            else:
                next_state_array = np.array([next_state[key] for key in feature_keys])

            # 更新緩衝區
            state_buffer.pop(0)
            state_buffer.append(next_state_array)

            episode_reward += reward
            state = next_state

            if done:
                break

        total_reward += episode_reward

    return total_reward / episodes



def check_state_representation(encoder):
    """
    检查编码器的状态表示

    参数:
        encoder: 编码器模型

    返回:
        dict: 包含状态表示检查结果的字典
    """
    results = {}

    # 1. 检查模型参数
    results['parameter_check'] = {
        'total_parameters': encoder.count_params(),
        'trainable_parameters': sum([tf.keras.backend.count_params(w) for w in encoder.trainable_weights]),
    }

    # 2. 创建测试输入
    batch_size, seq_len, input_dim = 5, 10, 32
    test_input = tf.random.normal((batch_size, seq_len, input_dim))

    # 3. 前向传播获取状态表示
    try:
        # 尝试获取状态表示
        state_representation = encoder(test_input, training=False)
    except Exception as e:
        results['error'] = f"前向传播失败: {str(e)}"
        return results

    # 4. 分析状态表示
    results['state_analysis'] = {
        'shape': state_representation.shape.as_list(),
        'mean': float(tf.reduce_mean(state_representation).numpy()),
        'std': float(tf.math.reduce_std(state_representation).numpy()),
        'min': float(tf.reduce_min(state_representation).numpy()),
        'max': float(tf.reduce_max(state_representation).numpy()),
        'has_nan': tf.reduce_any(tf.math.is_nan(state_representation)).numpy(),
        'has_inf': tf.reduce_any(tf.math.is_inf(state_representation)).numpy(),
    }

    # 5. 检查状态表示的多样性（避免模式崩溃）
    if len(state_representation.shape) > 1:
        # 计算不同样本之间的余弦相似度
        flattened_states = tf.reshape(state_representation, [state_representation.shape[0], -1])
        norms = tf.norm(flattened_states, axis=1, keepdims=True)
        normalized_states = flattened_states / norms
        cosine_similarity = tf.matmul(normalized_states, normalized_states, transpose_b=True)

        # 排除对角线元素（自相似性）
        mask = 1 - tf.eye(cosine_similarity.shape[0], dtype=cosine_similarity.dtype)
        avg_similarity = tf.reduce_sum(cosine_similarity * mask) / tf.reduce_sum(mask)

        results['diversity_analysis'] = {
            'avg_cosine_similarity': float(avg_similarity.numpy()),
            'is_diverse': float(avg_similarity.numpy()) < 0.8  # 阈值可根据实际情况调整
        }

    # 6. 检查梯度信息
    test_input_var = tf.Variable(test_input)
    with tf.GradientTape() as tape:
        output = encoder(test_input_var, training=True)
        loss = tf.reduce_sum(output)

    # 计算梯度
    grads = tape.gradient(loss, encoder.trainable_variables)

    # 检查梯度是否存在
    has_gradients = any(grad is not None for grad in grads)
    gradient_norms = [tf.norm(grad).numpy() for grad in grads if grad is not None]

    results['gradient_analysis'] = {
        'has_gradients': has_gradients,
        'gradient_norms': gradient_norms,
        'total_gradient_norm': sum(gradient_norms) if has_gradients else 0
    }

    return results


def check_exploration(qmodel ):
    """
    檢查探索狀態和統計信息

    Args:
        detailed: 是否顯示詳細信息

    Returns:
        探索狀態統計字典
    """
    total = qmodel.exploration_stats['total_choices']
    if total == 0:
        return {"status": "尚未進行任何選擇"}

    exploration_rate = (qmodel.exploration_stats['exploration_choices'] / total) * 100
    exploitation_rate = (qmodel.exploration_stats['exploitation_choices'] / total) * 100

    stats = {
        'total_choices': total,
        'exploration_choices': qmodel.exploration_stats['exploration_choices'],
        'exploitation_choices': qmodel.exploration_stats['exploitation_choices'],
        'exploration_rate_percent': round(exploration_rate, 2),
        'exploitation_rate_percent': round(exploitation_rate, 2),
        'current_epsilon': round(qmodel.epsilon, 4),
        'epsilon_min': qmodel.epsilon_min,
        'last_choice_type': qmodel.exploration_stats['last_choice_type'],
        'is_exploring': exploration_rate > 20  # 如果探索率大於20%，認為還在探索階段
    }
    return stats

def check_training_process(qmodel, detailed=False, plot=False):
    """
    檢查訓練過程狀態

    Args:
        detailed: 是否顯示詳細信息
        plot: 是否生成訓練曲線圖

    Returns:
        訓練過程統計字典
    """
    if not qmodel.training_history['episodes']:
        return {"status": "尚未開始訓練"}

    current_episode = qmodel.current_episode
    latest_reward = qmodel.training_history['rewards'][-1] if qmodel.training_history['rewards'] else 0
    avg_reward = qmodel.training_history['avg_rewards'][-1] if qmodel.training_history['avg_rewards'] else 0

    stats = {
        'current_episode': current_episode,
        'total_episodes': len(qmodel.training_history['episodes']),
        'latest_reward': round(latest_reward, 2),
        'latest_avg_reward': round(avg_reward, 2),
        'current_epsilon': round(qmodel.epsilon, 4),
        'q_table_size': len(qmodel.q_table),
        'training_duration': qmodel._get_training_duration(),
        'overall_avg_reward': round(np.mean(qmodel.training_history['rewards']), 2) if qmodel.training_history[
            'rewards'] else 0,'best_reward': round(max(qmodel.training_history['rewards']), 2) if qmodel.training_history['rewards'] else 0,
        'convergence_status': qmodel._check_convergence()
    }

    if detailed:
        # 添加詳細統計
        stats.update({
            'reward_trend': qmodel._get_reward_trend(),
            'exploration_trend': qmodel._get_exploration_trend(),
            'recent_performance': qmodel._get_recent_performance(10),
            'action_distribution': qmodel._get_current_action_distribution()
        })

    if plot:
        qmodel._plot_training_progress()

    return stats



def comprehensive_debug_checklist(qmodel, env, encoder):
    """
    完整的调试检查清单
    """
    print("=" * 60)
    print("🤖 强化学习算法调试检查清单")
    print("=" * 60)

    results = {}

    # 1. 检查奖励函数
    print("\n1. 🎯 奖励函数检查:")
    results['reward_function'] = check_reward_function(qmodel)

    # 2. 检查Q值合理性
    print("\n2. 📊 Q值合理性检查:")
    results['q_value_sanity'] = check_q_value_sanity(qmodel, encoder)

    # 3. 检查探索策略
    print("\n3. 🔍 探索策略检查:")
    results['exploration_check'] = check_exploration(qmodel)

    # 4. 检查神经网络训练
    print("\n4. 🧠 神经网络训练检查:")
    results['training_check'] = check_training_process(qmodel)

    # 5. 检查状态表示
    print("\n5. 📋 状态表示检查:")
    results['state_representation'] = check_state_representation(encoder)

    return results


def check_reward_function(qmodel):
    """检查奖励函数"""
    test_cases = [
        # (label, bits, expected_reward)
        (0, [0, 0, 0, 0], "high"),  # 所有关闭应该高奖励
        (1, [1, 1, 1, 1], "low"),  # 所有开启应该低奖励
        (0, [1, 0, 0, 0], "medium"),  # 适度使用
    ]

    for label, bits, expectation in test_cases:
        reward = qmodel.compute_reward_batch(
            np.array([label]),
            np.array([bits])
        )[0]
        print(f"  标签 {label}, 动作 {bits} → 奖励: {reward:.3f} ({expectation})")

    return "Reward function check completed"






def check_q_value_sanity(qmodel, encoder):
    """
    检查Q值的合理性

    参数:
        qmodel: Q值模型
        encoder: 编码器模型

    返回:
        dict: 包含Q值检查结果的字典
    """
    results = {}

    # 1. 创建测试输入（注意匹配编码器期望的输入形状）
    batch_size, seq_len, input_dim = 1, 10, 7  # 根据错误信息调整

    # 创建符合编码器输入形状的测试数据
    test_states = tf.random.normal((batch_size, seq_len, input_dim))

    # 2. 获取状态表示
    try:
        with tf.device('/CPU:0'):  # 避免GPU内存问题
            #state_sequence = np.array(state_buffer).reshape(1, window_size, -1)
            #state_latent = encoder(state_sequence).numpy()
            state_representations = encoder(test_states, training=False)
        results['encoder_output_shape'] = state_representations.shape.as_list()
    except Exception as e:
        results['encoder_error'] = f"编码器前向传播失败: {str(e)}"
        return results

    # 3. 检查Q值模型输入兼容性
    try:
        # 创建符合Q值模型输入形状的测试数据
        if len(qmodel.inputs) > 0:
            q_input_shape = qmodel.inputs[0].shape.as_list()
            results['q_model_input_shape'] = q_input_shape

            # 确保状态表示与Q模型输入兼容
            if len(state_representations.shape) == len(q_input_shape):
                # 调整状态表示形状以匹配Q模型输入
                adjusted_states = state_representations
                if state_representations.shape[1:] != q_input_shape[1:]:
                    # 如果需要重塑
                    adjusted_states = tf.reshape(
                        state_representations,
                        [batch_size] + q_input_shape[1:]
                    )

                # 获取Q值
                q_values = qmodel(adjusted_states, training=False)
                results['q_values_shape'] = q_values.shape.as_list()

                # 4. 分析Q值
                results['q_value_analysis'] = {
                    'mean': float(tf.reduce_mean(q_values).numpy()),
                    'std': float(tf.math.reduce_std(q_values).numpy()),
                    'min': float(tf.reduce_min(q_values).numpy()),
                    'max': float(tf.reduce_max(q_values).numpy()),
                    'range': float(tf.reduce_max(q_values).numpy() - tf.reduce_min(q_values).numpy()),
                    'has_nan': tf.reduce_any(tf.math.is_nan(q_values)).numpy(),
                    'has_inf': tf.reduce_any(tf.math.is_inf(q_values)).numpy(),
                }

                # 5. 检查Q值是否过于极端
                q_values_flat = tf.reshape(q_values, [-1])
                extreme_values = tf.reduce_sum(
                    tf.cast(tf.abs(q_values_flat) > 100.0, tf.float32)
                ) / tf.cast(tf.size(q_values_flat), tf.float32)

                results['extreme_value_analysis'] = {
                    'extreme_value_ratio': float(extreme_values.numpy()),
                    'has_extreme_values': float(extreme_values.numpy()) > 0.1
                }

            else:
                results[
                    'shape_mismatch'] = f"形状不匹配: 编码器输出 {state_representations.shape}, Q模型输入 {q_input_shape}"
        else:
            results['q_model_error'] = "无法获取Q模型的输入形状"

    except Exception as e:
        results['q_model_error'] = f"Q模型前向传播失败: {str(e)}"

    # 6. 检查梯度计算
    try:
        test_states_var = tf.Variable(test_states)
        with tf.GradientTape() as tape:
            state_repr = encoder(test_states_var, training=True)
            if len(state_repr.shape) == len(qmodel.inputs[0].shape):
                adjusted_states = tf.reshape(
                    state_repr,
                    [batch_size] + qmodel.inputs[0].shape[1:].as_list()
                )
                q_vals = qmodel(adjusted_states, training=True)
                loss = tf.reduce_mean(q_vals)

                grads = tape.gradient(loss, qmodel.trainable_variables)
                has_gradients = any(grad is not None for grad in grads)

                results['gradient_check'] = {
                    'has_gradients': has_gradients,
                    'gradient_norms': [float(tf.norm(grad).numpy()) for grad in grads if grad is not None]
                }
    except Exception as e:
        results['gradient_error'] = f"梯度计算失败: {str(e)}"

    return results

import csv

class TrainingMonitor:
    def __init__(self, save_path=None):
        self.episodes = []
        self.rewards = []
        self.avg_qs = []
        self.save_path = save_path

    def record_episode(self, episode, reward, avg_q):
        """记录一次训练的结果"""
        self.episodes.append(episode)
        self.rewards.append(reward)
        self.avg_qs.append(avg_q)

        print(f"[Monitor] Episode {episode} | Reward={reward:.2f} | AvgQ={avg_q:.2f}")

    def save_to_csv(self, filename=None):
        """保存结果到 CSV 文件"""
        if filename is None:
            if self.save_path:
                filename = self.save_path
            else:
                filename = "training_log.csv"

        with open(filename, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Episode", "Reward", "AvgQ"])
            for e, r, q in zip(self.episodes, self.rewards, self.avg_qs):
                writer.writerow([e, r, q])
        print(f"[Monitor] Results saved to {filename}")

    def plot(self, show=True, save_path=None):
        """绘制 reward 和 avg_q 曲线"""
        plt.figure(figsize=(10, 5))

        plt.subplot(1, 2, 1)
        plt.plot(self.episodes, self.rewards, label="Reward")
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.title("Reward per Episode")
        plt.grid(True)
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(self.episodes, self.avg_qs, label="Avg Q-value", color="orange")
        plt.xlabel("Episode")
        plt.ylabel("Avg Q")
        plt.title("Average Q per Episode")
        plt.grid(True)
        plt.legend()

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
            print(f"[Monitor] Plot saved to {save_path}")
        if show:
            plt.show()

def diagnostic_training(model, encoder, env, monitor, episodes=50):
    """诊断训练过程 - 修復NoneType錯誤版本"""
    print("開始診斷訓練...")

    for episode in range(episodes):
        try:
            # 環境重置
            reset_result = env.reset()
            print(f"重置結果類型: {type(reset_result)}")

            # 解析重置結果
            state, info = _parse_env_result(reset_result)
            print(f"解析後狀態類型: {type(state)}, 值: {state}")

            # 檢查並清理狀態數據
            state = _clean_and_validate_state(state)
            print(f"清理後狀態: {state}")

            total_reward = 0
            episode_q_values = []
            done = False
            step = 0

            while not done and step < 100:
                try:
                    # 檢查狀態有效性
                    if state is None:
                        print("狀態為None，跳過此步驟")
                        break

                    # 重塑狀態並檢查形狀
                    state_reshaped = _safe_reshape(state)
                    print(f"重塑後狀態形狀: {state_reshaped.shape}")

                    # 使用編碼器獲取潛在表示
                    state_latent = encoder(state_reshaped).numpy()
                    print(f"潛在狀態形狀: {state_latent.shape}")

                    # 選擇動作
                    action = model.choose_action(state_latent)
                    print(f"選擇動作: {action}")

                    # 執行動作
                    step_result = env.step(action)
                    next_state, reward, done, info = _parse_env_result(step_result)

                    # 清理下一個狀態
                    next_state = _clean_and_validate_state(next_state)
                    print(f"下一步狀態: {next_state}")

                    # 獲取下一個潛在狀態
                    next_state_reshaped = _safe_reshape(next_state)
                    next_state_latent = encoder(next_state_reshaped).numpy()

                    # 選擇下一個動作
                    next_action = model.choose_action(next_state_latent)

                    # 更新模型
                    model.update(state_latent, action, reward, next_state_latent, next_action)

                    # 記錄數據
                    total_reward += reward
                    q_values = model.get_q_values(state_latent)
                    episode_q_values.append(q_values)

                    # 更新狀態
                    state = next_state
                    step += 1

                    print(f"步驟 {step}: 獎勵={reward}, 總獎勵={total_reward}")

                    # 監控
                    if monitor and step % 10 == 0:
                        monitor.record_step(episode, step, reward, q_values, action)

                except Exception as e:
                    print(f"步驟 {step} 出錯: {e}")
                    import traceback
                    traceback.print_exc()
                    break

            # 記錄回合
            if monitor:
                avg_q = np.mean(episode_q_values) if episode_q_values else 0
                monitor.record_episode(episode, total_reward, avg_q)

            # 診斷信息
            print(f"Episode {episode}: 總獎勵={total_reward}, 步數={step}")
            _print_diagnostic_info(model, episode, total_reward, step)

        except Exception as e:
            print(f"回合 {episode} 初始化出錯: {e}")
            import traceback
            traceback.print_exc()
            continue


def _clean_and_validate_state(state):
    """清理和驗證狀態數據"""
    if state is None:
        print("警告: 狀態為None，使用默認狀態")
        return np.zeros(1)  # 返回默認狀態

    # 如果狀態是字典，嘗試提取數值
    if isinstance(state, dict):
        print("狀態是字典，嘗試提取數值...")
        # 嘗試找到包含數值的鍵
        for key, value in state.items():
            if value is not None and isinstance(value, (int, float, np.ndarray, list)):
                print(f"使用鍵 '{key}' 的值: {value}")
                state = value
                break
        else:
            # 如果沒有找到合適的值，使用所有值
            values = list(state.values())
            if all(v is not None for v in values):
                state = values
            else:
                print("字典中包含None值，使用默認狀態")
                return np.zeros(1)

    # 轉換為numpy數組
    if not isinstance(state, np.ndarray):
        try:
            state = np.array(state, dtype=np.float32)
        except Exception as e:
            print(f"轉換為數組失敗: {e}, 使用默認狀態")
            return np.zeros(1)

    # 檢查是否包含None值
    if np.any(state == None):  # noqa: E711
        print("狀態中包含None值，進行清理...")
        # 將None替換為0
        state = np.where(state == None, 0, state)  # noqa: E711

    # 檢查NaN值
    if np.any(np.isnan(state)):
        print("狀態中包含NaN值，進行清理...")
        state = np.nan_to_num(state)

    return state


def _safe_reshape(state):
    """安全地重塑狀態"""
    # 確保是numpy數組
    if not isinstance(state, np.ndarray):
        state = np.array(state)

    # 檢查維度
    if state.ndim == 1:
        return state.reshape(1, -1)
    elif state.ndim == 2:
        return state
    else:
        print(f"不支持的狀態維度: {state.ndim}, 嘗試展平")
        return state.flatten().reshape(1, -1)


def _parse_env_result(env_result):
    """解析環境返回結果 - 加強版本"""
    print(f"解析環境結果: {type(env_result)}")

    if env_result is None:
        print("環境返回None，使用默認值")
        return np.zeros(1), {}

    if isinstance(env_result, tuple):
        print(f"元組長度: {len(env_result)}")
        if len(env_result) == 2:
            state, info = env_result
            return state, info if isinstance(info, dict) else {}
        elif len(env_result) == 4:
            state, reward, done, info = env_result
            return state, info
        else:
            print(f"未知的元組長度: {len(env_result)}")
            return env_result[0], {}

    elif isinstance(env_result, dict):
        print(f"字典鍵: {list(env_result.keys())}")
        state = env_result.get('observation',
                               env_result.get('state',
                                              env_result.get('obs',
                                                             None)))

        if state is None:
            # 嘗試找到第一個合適的值
            for key, value in env_result.items():
                if value is not None and not isinstance(value, (str, dict, list)):
                    state = value
                    break
            else:
                state = np.zeros(1)

        info = {k: v for k, v in env_result.items()
                if k not in ['observation', 'state', 'obs', 'reward', 'done']}
        return state, info

    else:
        print(f"單一返回值: {env_result}")
        return env_result, {}


def _print_diagnostic_info(model, episode, reward, steps):
    """打印診斷信息"""
    try:
        stats = model.check_training_process(detailed=False)
        print(f"  探索率: {stats.get('exploration_rate_percent', 0):.1f}%")
        print(f"  Q-table大小: {stats.get('q_table_size', 0)}")
    except Exception as e:
        print(f"獲取診斷信息失敗: {e}")



def _ensure_array_format(data):
    """確保數據是numpy數組格式"""
    if isinstance(data, np.ndarray):
        return data
    elif isinstance(data, (list, tuple)):
        return np.array(data)
    elif isinstance(data, dict):
        # 嘗試從字典中提取數值數據
        numeric_values = [v for v in data.values() if isinstance(v, (int, float))]
        if numeric_values:
            return np.array(numeric_values)
        else:
            return np.array(list(data.values()))
    else:
        return np.array([data])




class RL_Debugger:
    def __init__(self):
        self.history = {
            'rewards': [], 'q_values': [], 'actions': [],
            'losses': [], 'epsilons': [], 'correlations': []
        }

    def log_episode(self, rewards, q_values, actions, loss, epsilon):
        self.history['rewards'].append(np.mean(rewards))
        self.history['q_values'].append(np.mean(q_values))
        self.history['actions'].extend(actions)
        self.history['losses'].append(loss)
        self.history['epsilons'].append(epsilon)

        # 计算相关性
        if len(rewards) > 1 and len(q_values) > 1:
            corr = np.corrcoef(rewards, q_values)[0, 1]
            self.history['correlations'].append(corr)

    def plot_diagnostics(self):
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))

        # 1. 奖励趋势
        axes[0, 0].plot(self.history['rewards'])
        axes[0, 0].set_title('Average Reward per Episode')
        axes[0, 0].set_ylabel('Reward')

        # 2. Q值趋势
        axes[0, 1].plot(self.history['q_values'])
        axes[0, 1].set_title('Average Q-value per Episode')
        axes[0, 1].set_ylabel('Q-value')

        # 3. 损失函数
        axes[1, 0].plot(self.history['losses'])
        axes[1, 0].set_title('Training Loss')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].set_yscale('log')  # 对数尺度看变化

        # 4. 探索率
        axes[1, 1].plot(self.history['epsilons'])
        axes[1, 1].set_title('Exploration Rate (Epsilon)')
        axes[1, 1].set_ylabel('Epsilon')

        # 5. 动作分布
        action_counts = pd.Series(self.history['actions']).value_counts()
        axes[2, 0].bar(action_counts.index, action_counts.values)
        axes[2, 0].set_title('Action Distribution')
        axes[2, 0].set_xlabel('Action')
        axes[2, 0].set_ylabel('Count')

        # 6. 相关性趋势
        axes[2, 1].plot(self.history['correlations'])
        axes[2, 1].set_title('Reward-Q Correlation')
        axes[2, 1].set_ylabel('Correlation')
        axes[2, 1].axhline(y=0, color='r', linestyle='--')

        plt.tight_layout()
        plt.show()


def run_comprehensive_debug(qmodel, encoder, env, data_files):
    """
    执行完整的调试流程
    """
    print("开始全面调试...")

    # 1. 运行基本检查
    checklist_results = comprehensive_debug_checklist(qmodel, env, encoder)

    # 2. 收集训练数据
    monitor = create_training_monitor()

    # 3. 进行短期训练并监控
    print("\n进行诊断训练...")
    monitor = TrainingMonitor(save_path="training_results.csv")

    diagnostic_training(qmodel, encoder, env, monitor, episodes=50)

    # 4. 分析结果
    #print("\n分析调试结果...")
    #analyze_debug_results(monitor, checklist_results)

    #5. 提供调试建议
    #provide_debug_recommendations(monitor)


# 使用示例
#if __name__ == "__main__":
# 初始化你的组件
# qmodel = YourDQNAgent(...)
# encoder = YourEncoder(...)
# env = YourEnvironment(...)

# 运行调试
# run_comprehensive_debug(qmodel, encoder, env, files)

# 或者单独运行某些调试功能
#debugger = RL_Debugger()
#hyperparameter_tuning(qmodel, encoder, env)

