import numpy as np
import itertools


# ---------------- Reward 计算 ----------------
def compute_reward_batch(labels, actions_bits, prev_actions_bits,
                         energy_penalty=0.1, match_bonus=0.5, switch_penalty_per_toggle=0.2):
    rewards = []
    for t in range(len(labels)):
        reward = 0.0

        # labels 假设 [ac, dehum]
        label_ac, label_dehum = labels[t][0], labels[t][1]
        act_ac, act_dehum = actions_bits[t][0], actions_bits[t][2]  # [ac, heater, dehum, hum]

        # 状态一致奖励
        if act_ac == label_ac:
            reward += match_bonus
        if act_dehum == label_dehum:
            reward += match_bonus

        # 能耗惩罚（所有设备）
        energy_cost = np.sum(actions_bits[t])
        reward -= energy_penalty * energy_cost

        # 开关惩罚（仅 ac, dehum）
        if t > 0:
            prev_ac, prev_dehum = prev_actions_bits[t][0], prev_actions_bits[t][2]
            if act_ac != prev_ac:
                reward -= switch_penalty_per_toggle
            if act_dehum != prev_dehum:
                reward -= switch_penalty_per_toggle

        rewards.append(reward)
    return np.array(rewards)


# ---------------- 环境 ----------------
class HVACEnv:
    def __init__(self, labels, reward_params):
        self.labels = labels
        self.reward_params = reward_params
        self.t = 0
        self.prev_action = np.zeros(4)  # [ac, heater, dehum, hum]

    def reset(self):
        self.t = 0
        self.prev_action = np.zeros(4)
        return self.labels[self.t]

    def step(self, action_bits):
        label = self.labels[self.t]
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


# ---------------- 假数据 ----------------
np.random.seed(42)
T = 30
labels = np.random.randint(0, 2, size=(T, 2))  # [ac, dehum] ground truth
actions = np.random.randint(0, 2, size=(T, 4))  # [ac, heater, dehum, hum]

# ---------------- 网格搜索 ----------------
param_grid = {
    "energy_penalty": [0.05, 0.1],
    "match_bonus": [0.5, 1.0],
    "switch_penalty_per_toggle": [0.1, 0.2],
}

for energy_penalty, match_bonus, switch_penalty in itertools.product(
        param_grid["energy_penalty"], param_grid["match_bonus"], param_grid["switch_penalty_per_toggle"]):

    reward_params = {
        "energy_penalty": energy_penalty,
        "match_bonus": match_bonus,
        "switch_penalty_per_toggle": switch_penalty,
    }
    env = HVACEnv(labels, reward_params)

    state = env.reset()
    total_reward = 0
    for t in range(T):
        _, r, done, _ = env.step(actions[t])
        total_reward += r
        if done:
            break

    print(f"Params {reward_params} => Total Reward = {total_reward:.2f}")
