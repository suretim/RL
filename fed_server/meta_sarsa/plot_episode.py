import matplotlib.pyplot as plt

# 你的 log 數據
episodes = list(range(1, 201))
eps = [0.298,0.285,0.271,0.258,0.245,0.233,0.222,0.211,0.201,0.191,
       0.182,0.173,0.164,0.156,0.149,0.141,0.135,0.128,0.122,0.116,0.110]
loss = [4.9084,4.9727,4.8470,4.9186,4.8839,4.8917,4.8858,4.9225,4.9076,4.8433,
        4.7919,4.9017,4.8544,4.7827,4.9252,4.8960,4.9090,4.9586,4.8479,4.8264,4.8606]

# 這裡 eps/loss 只有每10回合記錄一次 → 我們補成對應的 episode
ep_idx = list(range(1, 201, 10)) + [200]

plt.figure(figsize=(10,4))

# Loss 曲線
plt.subplot(1,2,1)
plt.plot(ep_idx, loss, marker='o')
plt.xlabel("Episode")
plt.ylabel("Loss")
plt.title("Loss vs Episode")

# Epsilon 曲線
plt.subplot(1,2,2)
plt.plot(ep_idx, eps, marker='o', color='orange')
plt.xlabel("Episode")
plt.ylabel("Epsilon")
plt.title("Epsilon vs Episode")

plt.tight_layout()
plt.show()
