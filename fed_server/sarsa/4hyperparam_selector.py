#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HVAC Q-Learning 完整训练与部署示例
- 使用 Optuna 或手动找到的最佳参数
- 支持训练、推理、保存与加载
"""

import numpy as np
import pickle
from utils_module import *

# -------------------------------
# 使用最佳参数训练
# -------------------------------
BEST = {
    "energy_penalty": 0.05,
    "match_bonus": 0.7,
    "switch_penalty_per_toggle": 0.3,
}

def train_final_agent(n_episodes=300, seq_len=20):
    env = PlantHVACEnv(seq_len=seq_len)
    agent = QLearningAgent(alpha=0.1, gamma=0.9, epsilon=0.3)  # 初期探索
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

# -------------------------------
# 推理 / rollout
# -------------------------------
def rollout(agent, seq_len=20):
    env = PlantHVACEnv(seq_len=seq_len)
    state = env.reset()
    done = False
    traj = []
    while not done:
        old_eps = agent.epsilon
        agent.epsilon = 0.0  # 贪心
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

# -------------------------------
# 保存 / 加载策略
# -------------------------------
def save_policy(agent, best_params, path="hvac_q_policy.pkl"):
    payload = {"q_table": agent.q_table, "params": best_params}
    with open(path, "wb") as f:
        pickle.dump(payload, f)

def load_policy(path="hvac_q_policy.pkl"):
    with open(path, "rb") as f:
        payload = pickle.load(f)
    agent = QLearningAgent()
    agent.q_table = payload["q_table"]
    best_params = payload["params"]
    return agent, best_params

# -------------------------------
# 主流程示例
# -------------------------------
if __name__ == "__main__":
    final_agent, avg_reward = train_final_agent()
    print("Final avg reward:", avg_reward)

    trajectory = rollout(final_agent, seq_len=20)
    print("First 5 steps of rollout:")
    for step in trajectory[:5]:
        print(step)

    save_policy(final_agent, BEST)
    loaded_agent, loaded_params = load_policy()
    print("Loaded params:", loaded_params)
