import optuna
import numpy as np
import itertools

# -------------------------------
# 你的 reward 计算函数
# -------------------------------
def compute_reward_sequence(labels, actions_bits,
                            healthy_state=0,
                            energy_penalty=0.1,
                            match_bonus=0.5,
                            switch_penalty_per_toggle=0.2,
                            prev_actions=None):
    labels = np.asarray(labels)
    actions_bits = np.asarray(actions_bits)
    T = labels.shape[0]

    health_state = labels[:, 0]
    hvac_targets = labels[:, 1:3]

    health_score = np.where(health_state == healthy_state, 1.0, -1.0)
    energy_cost = energy_penalty * np.sum(actions_bits, axis=1)
    match_score = np.sum((hvac_targets == 1) & (actions_bits == 1), axis=1) * match_bonus

    toggles = np.zeros(T, dtype=float)
    if prev_actions is not None:
        toggles[0] = np.sum(np.abs(actions_bits[0] - prev_actions))
    if T >= 2:
        diffs = np.abs(actions_bits[1:] - actions_bits[:-1])
        toggles[1:] = np.sum(diffs, axis=1)
    switch_penalty = switch_penalty_per_toggle * toggles

    return health_score - energy_cost + match_score - switch_penalty


# -------------------------------
# 模拟环境 (示例环境)
# -------------------------------
class PlantHVACEnv:
    def __init__(self, seq_len=20):
        self.seq_len = seq_len
        self.reset()

    def reset(self):
        # 随机生成 labels: [health, ac_target, dehum_target]
        self.labels_full = np.zeros((self.seq_len+1, 3), dtype=int)
        self.labels_full[:,0] = np.random.choice([0,1], size=self.seq_len+1)      # 健康/生病
        self.labels_full[:,1] = np.random.choice([0,1], size=self.seq_len+1)      # AC target
        self.labels_full[:,2] = np.random.choice([0,1], size=self.seq_len+1)      # Dehum target
        self.t = 0
        return self._get_state()

    def _get_state(self):
        return self.labels_full[self.t]   # 当前的状态信息

    def step(self, action, params):
        """
        action: [ac, dehum] 0/1
        params: dict 包含 reward 函数参数
        """
        next_label = self.labels_full[self.t+1]  # 下一时刻的 label
        reward = compute_reward_sequence(
            labels=np.expand_dims(next_label,0),
            actions_bits=np.expand_dims(action,0),
            **params
        )[0]
        self.t += 1
        done = (self.t >= self.seq_len)
        return self._get_state(), reward, done


# -------------------------------
# 简单 Q-learning Agent
# -------------------------------
class QLearningAgent:
    def __init__(self, n_actions=4, alpha=0.1, gamma=0.9, epsilon=0.2):
        self.n_actions = n_actions   # 4 种组合: [00,01,10,11]
        self.q_table = {}            # dict: key=state(tuple), value=np.array(Q-values)
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    def _state_to_key(self, state):
        return tuple(state.tolist())

    def select_action(self, state):
        key = self._state_to_key(state)
        if key not in self.q_table:
            self.q_table[key] = np.zeros(self.n_actions)
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)
        return np.argmax(self.q_table[key])

    def update(self, state, action, reward, next_state, done):
        key = self._state_to_key(state)
        if key not in self.q_table:
            self.q_table[key] = np.zeros(self.n_actions)
        if done:
            target = reward
        else:
            next_key = self._state_to_key(next_state)
            if next_key not in self.q_table:
                self.q_table[next_key] = np.zeros(self.n_actions)
            target = reward + self.gamma * np.max(self.q_table[next_key])
        self.q_table[key][action] += self.alpha * (target - self.q_table[key][action])


# -------------------------------
# 训练函数
# -------------------------------
def train_agent(energy_penalty, match_bonus, switch_penalty_per_toggle,
                n_episodes=50, seq_len=20):
    env = PlantHVACEnv(seq_len=seq_len)
    agent = QLearningAgent()
    params = {
        "energy_penalty": energy_penalty,
        "match_bonus": match_bonus,
        "switch_penalty_per_toggle": switch_penalty_per_toggle,
    }

    total_rewards = []
    for ep in range(n_episodes):
        state = env.reset()
        done = False
        ep_reward = 0
        while not done:
            action_id = agent.select_action(state)
            action_bits = np.array([action_id // 2, action_id % 2])  # 00,01,10,11
            next_state, reward, done = env.step(action_bits, params)
            agent.update(state, action_id, reward, next_state, done)
            ep_reward += reward
            state = next_state
        total_rewards.append(ep_reward)
    return np.mean(total_rewards)  # 返回平均 reward


# -------------------------------
# 网格搜索
# -------------------------------
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

for ep, mb, sp in param_combos:
    score = train_agent(ep, mb, sp, n_episodes=50, seq_len=20)
    print(f"Params ep={ep}, mb={mb}, sp={sp} => AvgReward={score:.2f}")
    if score > best_score:
        best_score = score
        best_params = (ep, mb, sp)

print("最佳参数:", best_params, "最佳平均reward:", best_score)


def objective(trial):
    energy_penalty = trial.suggest_uniform("energy_penalty", 0.01, 0.3)
    match_bonus = trial.suggest_uniform("match_bonus", 0.1, 1.0)
    switch_penalty = trial.suggest_uniform("switch_penalty_per_toggle", 0.05, 0.5)

    total_reward = train_agent(
        energy_penalty=energy_penalty,
        match_bonus=match_bonus,
        switch_penalty_per_toggle=switch_penalty
    )
    return total_reward


study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=30)

print("最佳参数:", study.best_params)
