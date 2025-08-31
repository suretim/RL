import matplotlib.pyplot as plt
import numpy as np

# === Rewards ===
rewards = [0.9,0.9,0.9,0.9,0.9,0.85,0.85,0.95,0.95,0.9,
           0.95,0.85,0.9,0.95,0.9,0.9,0.95,0.9,0.9,0.95,
           0.85,1.0,0.85,0.9,0.8,0.9,0.9,0.9,0.85,0.9]
episodes = np.arange(1, len(rewards)+1)

# 移動平均 (窗口=5)
window = 5
moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')

# === Actions ===
actions = [
 [0.0,0.0,1.0,1.0], [1.0,0.0,0.0,1.0], [0.0,0.0,1.0,1.0], [1.0,0.0,0.0,1.0],
 [1.0,0.0,1.0,0.0], [1.0,1.0,0.0,1.0], [1.0,0.0,1.0,1.0], [1.0,0.0,0.0,0.0],
 [0.0,0.0,1.0,0.0], [0.0,0.0,1.0,1.0], [0.0,1.0,0.0,0.0], [1.0,1.0,0.0,1.0],
 [0.0,1.0,1.0,0.0], [0.0,1.0,0.0,0.0], [0.0,1.0,0.0,1.0], [1.0,0.0,0.0,1.0],
 [0.0,0.0,1.0,0.0], [1.0,0.0,1.0,0.0], [0.0,1.0,1.0,0.0], [0.0,1.0,0.0,0.0],
 [1.0,0.0,1.0,1.0], [0.0,0.0,0.0,0.0], [1.0,1.0,0.0,1.0], [1.0,1.0,0.0,0.0],
 [1.0,1.0,1.0,1.0], [0.0,0.0,1.0,1.0], [0.0,0.0,1.0,1.0], [0.0,0.0,1.0,1.0],
 [0.0,1.0,1.0,1.0], [0.0,1.0,1.0,0.0]
]
actions = np.array(actions)

counts = np.sum(actions, axis=0)
zeros = len(actions) - counts
labels = [f"Action {i+1}" for i in range(actions.shape[1])]
x = np.arange(len(labels))

# === Plotting ===
fig, axes = plt.subplots(2, 1, figsize=(8,8))

# Reward 曲線
axes[0].plot(episodes, rewards, marker='o', label="Reward")
axes[0].plot(episodes[window-1:], moving_avg, label="Moving Avg (5)", linewidth=2, color='orange')
axes[0].set_xlabel("Episode")
axes[0].set_ylabel("Reward")
axes[0].set_title("Reward Trend")
axes[0].legend()
axes[0].grid(True, linestyle="--", alpha=0.6)

# Actions 分布
axes[1].bar(x-0.2, counts, width=0.4, label="ON (1)")
axes[1].bar(x+0.2, zeros, width=0.4, label="OFF (0)")
axes[1].set_xticks(x)
axes[1].set_xticklabels(labels)
axes[1].set_ylabel("Count")
axes[1].set_title("Action Switch Distribution")
axes[1].legend()
axes[1].grid(axis="y", linestyle="--", alpha=0.6)

plt.tight_layout()
plt.show()
