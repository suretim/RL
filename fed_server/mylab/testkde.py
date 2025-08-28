import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# 假设 good trial 的参数分布
good_params = np.array([0.2, 0.25, 0.3, 0.28, 0.35])
bad_params = np.array([0.6, 0.65, 0.7, 0.75, 0.8])

# KDE 拟合
kde_good = gaussian_kde(good_params)
kde_bad = gaussian_kde(bad_params)

# 画图
x = np.linspace(0,1,200)
plt.plot(x, kde_good(x), label="p(x|good)")
plt.plot(x, kde_bad(x), label="p(x|bad)")
plt.legend()
plt.title("KDE 拟合分布 (TPE 内部原理)")
plt.show()
