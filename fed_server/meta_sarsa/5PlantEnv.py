
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))  # RL 根目录
from global_hyparm import *
from utils_module import *

env = PlantHVACEnv(seq_len=20, n_features=3)
params = {"energy_penalty":0.1, "switch_penalty_per_toggle":0.2, "vpd_target":1.2, "vpd_penalty":2.0}

state = env.reset()
done = False
while not done:
    action = np.random.choice([0,1], size=4)  # 随机动作
    # 构造 dummy 序列输入 [1, seq_len, n_features]
    seq_input = np.random.rand(1,20,3).astype(np.float32)
    state, reward, done, info = env.step(action, seq_input, params)
    print(f"t={env.t}, action={action}, reward={reward:.3f}, flower_prob={info['flower_prob']:.2f}, temp={info['temp']:.1f}, humid={info['humid']:.2f}, vpd={info['vpd']:.2f}")
