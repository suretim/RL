import itertools
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ---------------- 套件函數 ----------------
def hvac_grid_search(labels, param_grid, episodes=50):
    """
    labels: [T,2] 健康狀態 (ac, dehum)
    param_grid: dict, 網格搜索參數
    episodes: 每組參數訓練輪數
    """
    # ---------------- 计算 reward ----------------
    def compute_reward_batch(labels, actions_bits, prev_actions_bits,
                             energy_penalty=0.1, match_bonus=0.5, switch_penalty_per_toggle=0.2):
        rewards = []
        for t in range(len(labels)):
            reward = 0.0
            label_ac, label_dehum = labels[t][0], labels[t][1]
            act_ac, act_dehum = actions_bits[t][0], actions_bits[t][2]

            if act_ac == label_ac:
                reward += match_bonus
            if act_dehum == label_dehum:
                reward += match_bonus

            energy_cost = np.sum(actions_bits[t])
            reward -= energy_penalty * energy_cost

            if t > 0:
                prev_ac, prev_dehum = prev_actions_bits[t][0], prev_actions_bits[t][2]
                if act_ac != prev_ac:
                    reward -= switch_penalty_per_toggle
                if act_dehum != prev_dehum:
                    reward -= switch_penalty_per_toggle

            rewards.append(reward)
        return np.array(rewards)

    # ---------------- 環境 ----------------
    class HVACEnv:
        def __init__(self, labels, reward_params):
            self.labels = labels
            self.reward_params = reward_params
            self.t = 0
            self.prev_action = np.zeros(4)
            self.action_space = [np.array(list(np.binary_repr(i, width=4)), dtype=int) for i in range(16)]

        def reset(self):
            self.t = 0
            self.prev_action = np.zeros(4)
            return self.labels[self.t]

        def step(self, action_idx):
            action_bits = self.action_space[action_idx]
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

    # ---------------- SARSA Agent ----------------
    class SarsaAgent:
        def __init__(self, action_dim, lr=0.1, gamma=0.99, epsilon=0.1):
            self.Q = np.zeros((2, 2, action_dim))
            self.lr = lr
            self.gamma = gamma
            self.epsilon = epsilon
            self.action_dim = action_dim

        def select_action(self, state):
            state_idx = (int(state[0]), int(state[1]))
            if np.random.rand() < self.epsilon:
                return np.random.randint(self.action_dim)
            return np.argmax(self.Q[state_idx])

        def update(self, state, action, reward, next_state, next_action, done):
            state_idx = (int(state[0]), int(state[1]))
            if done:
                target = reward
            else:
                next_idx = (int(next_state[0]), int(next_state[1]))
                target = reward + self.gamma * self.Q[next_idx][next_action]
            self.Q[state_idx][action] += self.lr * (target - self.Q[state_idx][action])

    # ---------------- 訓練函數 ----------------
    def train_agent(env, agent, episodes=50):
        total_reward = 0
        for ep in range(episodes):
            state = env.reset()
            action = agent.select_action(state)
            done = False
            while not done:
                next_state, reward, done, _ = env.step(action)
                next_action = agent.select_action(next_state) if not done else None
                agent.update(state, action, reward, next_state, next_action, done)
                total_reward += reward
                state, action = next_state, next_action
        return total_reward / episodes

    # ---------------- 網格搜索 ----------------
    param_combos = list(itertools.product(
        param_grid["energy_penalty"],
        param_grid["match_bonus"],
        param_grid["switch_penalty_per_toggle"]
    ))

    best_score, best_params = -1e9, None
    results = []

    for energy_penalty, match_bonus, switch_penalty in param_combos:
        reward_params = {
            "energy_penalty": energy_penalty,
            "match_bonus": match_bonus,
            "switch_penalty_per_toggle": switch_penalty,
        }
        env = HVACEnv(labels, reward_params)
        agent = SarsaAgent(action_dim=len(env.action_space))

        avg_reward = train_agent(env, agent, episodes)
        results.append((reward_params, avg_reward))
        print(f"Params {reward_params} => Avg Reward {avg_reward:.3f}")

        if avg_reward > best_score:
            best_score = avg_reward
            best_params = reward_params

    # ---------------- 3D 散點圖 ----------------
    xs, ys, zs, cs = [], [], [], []
    for (params, score) in results:
        xs.append(params["energy_penalty"])
        ys.append(params["match_bonus"])
        zs.append(score)
        cs.append(params["switch_penalty_per_toggle"])

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    scatter = ax.scatter(xs, ys, zs, c=cs, cmap="viridis", s=80)
    best_idx = zs.index(best_score)
    ax.scatter(xs[best_idx], ys[best_idx], zs[best_idx], color='red', s=150, label='Best Reward')
    ax.text(xs[best_idx], ys[best_idx], zs[best_idx], f"{zs[best_idx]:.2f}", color='red')
    ax.set_xlabel("Energy Penalty")
    ax.set_ylabel("Match Bonus")
    ax.set_zlabel("Average Reward")
    plt.title("Grid Search Results (color = switch_penalty)")
    plt.legend()
    fig.colorbar(scatter, label="Switch Penalty")
    plt.show()

    # ---------------- 2D Heatmap ----------------
    switch_fixed = best_params["switch_penalty_per_toggle"]
    energy_vals = sorted(list(set([r[0]["energy_penalty"] for r in results])))
    match_vals = sorted(list(set([r[0]["match_bonus"] for r in results])))
    reward_matrix = np.zeros((len(match_vals), len(energy_vals)))

    for (params, score) in results:
        if params["switch_penalty_per_toggle"] == switch_fixed:
            i = match_vals.index(params["match_bonus"])
            j = energy_vals.index(params["energy_penalty"])
            reward_matrix[i, j] = score

    plt.figure(figsize=(8, 6))
    im = plt.imshow(reward_matrix, origin='lower', cmap='YlGnBu', aspect='auto')
    best_energy_idx = energy_vals.index(best_params["energy_penalty"])
    best_match_idx = match_vals.index(best_params["match_bonus"])
    plt.scatter(best_energy_idx, best_match_idx, color='red', s=100, label='Best Reward')
    plt.text(best_energy_idx, best_match_idx, f"{best_score:.2f}", color='red', ha='center', va='bottom')
    plt.xticks(ticks=range(len(energy_vals)), labels=energy_vals)
    plt.yticks(ticks=range(len(match_vals)), labels=match_vals)
    plt.xlabel("Energy Penalty")
    plt.ylabel("Match Bonus")
    plt.title(f"Average Reward Heatmap (Switch Penalty = {switch_fixed})")
    plt.colorbar(im, label="Average Reward")
    plt.legend()
    plt.show()

    return best_params, best_score

# ---------------- 使用示例 ----------------
labels = np.random.randint(0, 2, size=(30, 2))
param_grid = {
    "energy_penalty": [0.05, 0.1, 0.2],
    "match_bonus": [0.3, 0.5, 0.7],
    "switch_penalty_per_toggle": [0.1, 0.2, 0.3],
}

best_params, best_score = hvac_grid_search(labels, param_grid, episodes=50)
print("✅ 最終最佳參數:", best_params, "最佳 reward:", best_score)
