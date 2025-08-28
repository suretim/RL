import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# 模拟 good 和 bad 样本
good_params = np.array([0.2, 0.25, 0.3, 0.28, 0.35])
bad_params = np.array([0.6, 0.65, 0.7, 0.75, 0.8])

# KDE 拟合
kde_good = gaussian_kde(good_params)
kde_bad = gaussian_kde(bad_params)

# 在区间 [0,1] 上评估
x = np.linspace(0, 1, 200)
px_good = kde_good(x)
px_bad = kde_bad(x)
ratio = px_good / (px_bad + 1e-9)  # 避免除零

# 画图
plt.figure(figsize=(8,5))
plt.plot(x, px_good, label="p(x|good)", color="blue")
plt.plot(x, px_bad, label="p(x|bad)", color="orange")
plt.plot(x, ratio/np.max(ratio), label="p(good)/p(bad) (归一化)", color="green", linestyle="--")
plt.legend()
plt.title("TPE: 蓝高橙低的地方 → 多采样")
plt.show()
