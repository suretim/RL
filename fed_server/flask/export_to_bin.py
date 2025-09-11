import numpy as np
import struct
import tensorflow as tf

# ====== 模拟 PPO 模型结构 ======
# 输入维度 = 5, 隐藏层 = 32, 动作维度 = 4
input_dim = 5
hidden_dim = 32
action_dim = 4
from util_hvac_agent import TensorFlowESP32Exporter


exporter = TensorFlowESP32Exporter(model_or_path="/tmp/model")

# 假设你已经有 Keras 模型 (Actor & Critic 共用 encoder)
inputs = tf.keras.Input(shape=(input_dim,))
x = tf.keras.layers.Dense(hidden_dim, activation="tanh")(inputs)

# Actor 输出 logits
actor_out = tf.keras.layers.Dense(action_dim)(x)

# Critic 输出 value
critic_out = tf.keras.layers.Dense(1)(x)

model = tf.keras.Model(inputs=inputs, outputs=[actor_out, critic_out])
model.summary()
# 假设这里载入训练好的权重
# model.load_weights("ppo_model.h5")

# ====== 提取参数 ======
# Dense 层顺序: [encoder, actor, critic]
encoder = model.layers[1]
actor = model.layers[2]
critic = model.layers[3]

W1, b1 = encoder.get_weights()   # W1.shape = (5,32), b1.shape = (32,)
W2, b2 = actor.get_weights()     # W2.shape = (32,4), b2.shape = (4,)
Vw, Vb = critic.get_weights()    # Vw.shape = (32,1), Vb.shape = (1,)

# ====== 存储为二进制 .bin ======
with open("saved_models/ppo_model.bin", "wb") as f:
    # W1 (input_dim * hidden_dim)
    f.write(W1.astype(np.float32).tobytes())
    # b1 (hidden_dim)
    f.write(b1.astype(np.float32).tobytes())
    # W2 (hidden_dim * action_dim)
    f.write(W2.astype(np.float32).tobytes())
    # b2 (action_dim)
    f.write(b2.astype(np.float32).tobytes())
    # Vw (hidden_dim)
    f.write(Vw.reshape(-1).astype(np.float32).tobytes())
    # Vb (1)
    f.write(Vb.astype(np.float32).tobytes())

print("✅ 导出完成: ppo_model.bin")
