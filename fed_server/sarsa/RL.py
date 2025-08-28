import itertools
import numpy as np

# ---------------- 假数据 ----------------
np.random.seed(42)
T = 30
labels = np.random.randint(0, 2, size=(T, 2))  # [ac, dehum] ground truth
actions = np.random.randint(0, 2, size=(T, 4)) # [ac, heater, dehum, hum]

def compute_reward_batch(labels, actions_bits, prev_actions_bits,
                         energy_penalty=0.1, match_bonus=0.5, switch_penalty_per_toggle=0.2):
    """
    计算批次奖励
    labels: [T, feat_dim] 健康状态 (只用到 ac, dehum 两列)
    actions_bits: [T, act_dim] HVAC 控制 bits (ac, heater, dehum, hum...)
    prev_actions_bits: [T, act_dim] 上一时刻的 HVAC bits（用来计算开关次数）

    返回: [T] reward 向量
    """
    rewards = []
    for t in range(len(labels)):
        reward = 0.0

        # ----------- 基础奖励：状态一致性 -----------
        label_ac, label_dehum = labels[t][0], labels[t][1]
        act_ac, act_dehum = actions_bits[t][0], actions_bits[t][2]  # 假设顺序是 [ac, heater, dehum, hum]

        if act_ac == label_ac:
            reward += match_bonus
        if act_dehum == label_dehum:
            reward += match_bonus

        # ----------- 能耗惩罚 -----------
        energy_cost = np.sum(actions_bits[t])  # 所有设备开的数量
        reward -= energy_penalty * energy_cost

        # ----------- 开关惩罚 (只针对 ac & dehum) -----------
        if t > 0:
            prev_ac, prev_dehum = prev_actions_bits[t][0], prev_actions_bits[t][2]
            if act_ac != prev_ac:
                reward -= switch_penalty_per_toggle
            if act_dehum != prev_dehum:
                reward -= switch_penalty_per_toggle

        rewards.append(reward)

    return np.array(rewards)


class HVACEnv:
    def __init__(self, labels, reward_params):
        self.labels = labels
        self.reward_params = reward_params
        self.t = 0
        self.prev_action = np.zeros(4)  # 假设有4个 hvac bits

    def reset(self):
        self.t = 0
        self.prev_action = np.zeros(4)
        return self.labels[self.t]

    def step(self, action_bits):
        label = self.labels[self.t]

        # 单步 reward
        reward = compute_reward_batch(
            labels=[label],
            actions_bits=[action_bits],
            prev_actions_bits=[self.prev_action],
            **self.reward_params
        )[0]

        self.prev_action = action_bits
        self.t += 1
        done = self.t >= len(self.labels)

        next_state = self.labels[self.t] if not done else None
        return next_state, reward, done, {}
reward_params = {
    "energy_penalty": 0.1,
    "match_bonus": 0.5,
    "switch_penalty_per_toggle": 0.2,
}

env = HVACEnv(labels, reward_params)


# -----------------------
# 假设你已经有的接口
# -----------------------
# env.reset() -> state
# env.step(action) -> next_state, reward, done, info
# agent.train(env, reward_params) -> 返回总 reward

def train_agent(env, agent, reward_params, episodes=50):
    """简单训练循环"""
    total_reward = 0
    for ep in range(episodes):
        state = env.reset()
        done = False
        while not done:
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action, reward_params)
            agent.update(state, action, reward, next_state, done)
            total_reward += reward
            state = next_state
    return total_reward / episodes  # 平均 reward


# -----------------------
# 网格搜索参数空间
# -----------------------
param_grid = {
    "energy_penalty": [0.05, 0.1, 0.2],
    "match_bonus": [0.3, 0.5, 0.7],
    "switch_penalty_per_toggle": [0.1, 0.2, 0.3],
}

param_combos = list(itertools.product(
    param_grid["energy_penalty"],
    param_grid["match_bonus"],
    param_grid["switch_penalty_per_toggle"]
))

best_score, best_params = -1e9, None
results = []


class SarsaAgent:
    def __init__(self, state_dim, action_dim, lr=0.1, gamma=0.99, epsilon=0.1):
        self.Q = np.zeros((state_dim, action_dim))
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.action_dim = action_dim

    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_dim)
        return np.argmax(self.Q[state])

    def update(self, state, action, reward, next_state, next_action):
        target = reward + self.gamma * self.Q[next_state, next_action]
        self.Q[state, action] += self.lr * (target - self.Q[state, action])

state = env.reset()  # state.shape -> (state_dim,)
state_dim = state.shape[0]
agent = SarsaAgent(state_dim= state_dim, action_dim=env.prev_action.shape[0])

# -----------------------
# 搜索过程
# -----------------------
for energy_penalty, match_bonus, switch_penalty in param_combos:
    reward_params = {
        "energy_penalty": energy_penalty,
        "match_bonus": match_bonus,
        "switch_penalty_per_toggle": switch_penalty,
    }

    # 这里调用训练过程
    avg_reward = train_agent(env, agent, reward_params)

    results.append((reward_params, avg_reward))
    print(f"Params {reward_params} => Avg Reward {avg_reward:.3f}")

    if avg_reward > best_score:
        best_score = avg_reward
        best_params = reward_params

print("✅ 最佳参数:", best_params, "最佳 reward:", best_score)

# -----------------------
# 可视化结果 (简单柱状图)
# -----------------------
'''
import matplotlib.pyplot as plt

labels = [f"ep={r[0]['energy_penalty']},mb={r[0]['match_bonus']},sp={r[0]['switch_penalty_per_toggle']}" for r in
          results]
scores = [r[1] for r in results]

plt.figure(figsize=(12, 6))
plt.bar(labels, scores)
plt.xticks(rotation=45, ha="right")
plt.ylabel("Average Reward")
plt.title("Grid Search Results")
plt.show()
'''