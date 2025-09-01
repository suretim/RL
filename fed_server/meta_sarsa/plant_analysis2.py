import time
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd


class QModel:
    def __init__(self, state_size=None, action_size=3, learning_rate=0.1, gamma=0.9,
                 initial_epsilon=0.3, config_file="best_params.json"):
        # ... 之前的初始化代碼 ...

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
            'best_reward': round(max(self.training_history['rewards']), 2) if self.training_history['rewards'] else 0,
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


# 使用示例
class TrafficAnalysis:
    def __init__(self):
        self.q_model = QModel(action_size=3)
        self.ACTIONS = {0: "NORMAL", 1: "LIMIT", 2: "BLOCK"}

    def train_episode(self, traffic_data_list):
        """訓練一個回合"""
        self.q_model.start_episode()

        for i, traffic_data in enumerate(traffic_data_list):
            state = self._extract_features(traffic_data)
            action = self.q_model.choose_action(state)
            reward = self._calculate_reward(traffic_data, action)
            next_state = self._extract_features(self._get_next_data())

            self.q_model.update(state, action, reward, next_state)

            # 每10步檢查一次訓練過程
            if i % 10 == 0:
                training_info = self.q_model.check_training_process()
                print(f"步驟 {i}: 獎勵={reward}, 探索率={training_info['current_epsilon']}")

        self.q_model.end_episode()

        # 每個回合結束後檢查
        if self.q_model.current_episode % 5 == 0:
            summary = self.q_model.get_training_summary()
            print(summary)

            # 每10個回合繪圖一次
            if self.q_model.current_episode % 10 == 0:
                self.q_model.check_training_process(plot=True)


# 快速測試
if __name__ == "__main__":
    model = QModel(action_size=3)

    # 模擬訓練過程
    for episode in range(1, 21):
        model.start_episode()

        # 模擬每個回合的10個步驟
        for step in range(10):
            state = [np.random.random() for _ in range(3)]
            action = model.choose_action(state)
            reward = np.random.randint(-5, 15)
            next_state = [np.random.random() for _ in range(3)]

            model.update(state, action, reward, next_state)

        model.end_episode()

        # 每5個回合檢查一次
        if episode % 5 == 0:
            stats = model.check_training_process(detailed=True)
            print(f"回合 {episode}: 獎勵={stats['latest_reward']}, 狀態數={stats['q_table_size']}")

    # 最終報告
    print("\n" + "=" * 50)
    print("最終訓練報告:")
    print(model.get_training_summary())
    model.check_training_process(plot=True)