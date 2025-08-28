import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from utils_module import *

# 假设 BEST 参数已经定义：
BEST = {
    "energy_penalty": 0.05,
    "match_bonus": 0.7,
    "switch_penalty_per_toggle": 0.3,
}

# 训练函数
def train_final_agent(n_episodes=300, seq_len=20):
    env = PlantHVACEnv(seq_len=seq_len)
    agent = QLearningAgent(alpha=0.1, gamma=0.9, epsilon=0.3)  # 初期多探索
    total_rewards = []

    for ep in range(n_episodes):
        agent.epsilon = max(0.02, 0.3 * (0.995 ** ep))  # ε 衰减
        state = env.reset()
        done = False
        ep_reward = 0.0
        while not done:
            action_id = agent.select_action(state)
            action_bits = np.array([action_id // 2, action_id % 2])
            next_state, reward, done = env.step(action_bits, BEST)
            agent.update(state, action_id, reward, next_state, done)
            ep_reward += reward
            state = next_state
        total_rewards.append(ep_reward)

    return agent, np.mean(total_rewards)

# 训练并生成 final_agent
final_agent, avg_reward = train_final_agent()
print("Final avg reward:", avg_reward)

def rollout(agent, seq_len=20):
    env = PlantHVACEnv(seq_len=seq_len)
    state = env.reset()
    done = False
    traj = []
    while not done:
        old_eps = agent.epsilon
        agent.epsilon = 0.0  # 贪心选择
        action_id = agent.select_action(state)
        action_bits = np.array([action_id // 2, action_id % 2])
        next_state, reward, done = env.step(action_bits, BEST)

        traj.append({
            "t": env.t,
            "state": state.tolist(),
            "action_id": int(action_id),
            "action_bits": action_bits.tolist(),
            "reward": float(reward)
        })

        state = next_state
        agent.epsilon = old_eps
    return traj

# 生成 trajectory
trajectory = rollout(final_agent, seq_len=20)

def visualize_rollout(trajectory):
    T = len(trajectory)
    rewards = [step['reward'] for step in trajectory]
    actions = np.array([step['action_bits'] for step in trajectory])  # shape [T,2]

    # 1️⃣ Reward 曲线
    plt.figure(figsize=(10, 4))
    plt.plot(range(T), rewards, marker='o', label="Reward per step")
    plt.xlabel("Time step")
    plt.ylabel("Reward")
    plt.title("Rollout Reward Curve")
    plt.grid(True)
    plt.legend()
    plt.show()

    # 2️⃣ 动作热力图
    plt.figure(figsize=(8, 4))
    sns.heatmap(actions.T, annot=True, cbar=True, cmap="YlGnBu", xticklabels=range(T), yticklabels=["AC", "Dehum"])
    plt.xlabel("Time step")
    plt.ylabel("Action Bits")
    plt.title("Actions Heatmap (1=ON, 0=OFF)")
    plt.show()


# 调用示例
visualize_rollout(trajectory)



def multi_rollout_visualize(agent, n_rollouts=50, seq_len=20):
    rewards_all = np.zeros((n_rollouts, seq_len))
    actions_all = np.zeros((n_rollouts, seq_len, 2))  # 2 个动作 bit: AC, Dehum

    for i in range(n_rollouts):
        env = PlantHVACEnv(seq_len=seq_len)
        state = env.reset()
        done = False
        t = 0
        while not done:
            old_eps = agent.epsilon
            agent.epsilon = 0.0  # 贪心选择
            action_id = agent.select_action(state)
            action_bits = np.array([action_id // 2, action_id % 2])
            next_state, reward, done = env.step(action_bits, BEST)

            rewards_all[i, t] = reward
            actions_all[i, t, :] = action_bits

            state = next_state
            agent.epsilon = old_eps
            t += 1

    # 平均 reward 曲线
    avg_rewards = rewards_all.mean(axis=0)
    plt.figure(figsize=(10, 4))
    plt.plot(range(seq_len), avg_rewards, marker='o')
    plt.xlabel("Time step")
    plt.ylabel("Average Reward")
    plt.title(f"Average Reward over {n_rollouts} Rollouts")
    plt.grid(True)
    plt.show()

    # 动作概率热力图
    action_probs = actions_all.mean(axis=0)  # shape [seq_len,2]
    plt.figure(figsize=(8, 4))
    sns.heatmap(action_probs.T, annot=True, cmap="YlGnBu", xticklabels=range(seq_len), yticklabels=["AC", "Dehum"],
                fmt=".2f")
    plt.xlabel("Time step")
    plt.ylabel("Action Probability")
    plt.title(f"Action Probability Heatmap over {n_rollouts} Rollouts")
    plt.show()


# 调用示例
multi_rollout_visualize(final_agent, n_rollouts=50, seq_len=20)
