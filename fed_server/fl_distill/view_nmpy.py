import numpy as np

#cache_train = "./kd_out/soft_train.npz"
cache_train = "./kd_out/soft_val.npz"

# 加载 .npz 文件
data = np.load(cache_train)

# 查看里面包含哪些数组
print("Keys:", data.files)

# 查看每个数组的形状和类型
for key in data.files:
    print(f"{key}: shape={data[key].shape}, dtype={data[key].dtype}")

# 如果想看前几个元素
print("x:",data["x"][:1])  # 图像
print("y:",data["y"][:1])  # 标签
print("soft:",data["soft"][:1])  # 教师 soft label
