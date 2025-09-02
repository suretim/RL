import tensorflow as tf
import numpy as np
import os
# -----------------------------
# 3sarsa_controller_full_fixed.py
# -----------------------------
from tensorflow.keras import layers, models, optimizers
from tensorflow import keras
from sklearn.model_selection import train_test_split
from utils_module import load_all_csvs
from global_hyparm import *
import pandas as pd
# ------------------ 配置 ------------------

ENCODER_LATENT_DIM = 16
Q_HIDDEN = [64, 64]

# FIX: 使用幂而不是按位异或；NUM_ACTIONS 由 NUM_SWITCH 决定
NUM_ACTIONS = 2 ** NUM_SWITCH  # e.g., NUM_SWITCH=4 -> 16

LR_Q = 1e-3
SARSA_EPISODES = 200
BATCH_MAX = 32
GAMMA = 0.95
EPS_START = 0.3
EPS_END = 0.1
EPS_DECAY = 0.995
ACTION_COST = 0.05

#from utils_fisher import *


class QModel:
    def __init__(self):
        self.q_net = self.build_q_network(dtype=tf.float32)
        self.q_optimizer = optimizers.Adam(LR_Q)
        self.mse_loss = tf.keras.losses.MeanSquaredError()



    def encode_sequence(self,encoder, X_seq):
        """
        Encode an entire sequence in batch.
        X_seq shape: (T, SEQ_LEN, NUM_FEATS) 或 (batch, SEQ_LEN, NUM_FEATS)
        Returns: (T, FEATURE_DIM)
        """
        if len(X_seq.shape) == 2:
            # 单样本 (SEQ_LEN, NUM_FEATS) -> (1, SEQ_LEN, NUM_FEATS)
            X_seq = np.expand_dims(X_seq, axis=0)

        # encoder 输出 (batch, FEATURE_DIM)
        z = encoder(X_seq).numpy()
        return z   # shape: (batch, FEATURE_DIM)

    # --------- Helper: 按时间步编码（不改 encoder 结构） ---------
    #@staticmethod
    def encode_timestep(self,encoder, X_seq, seq_len=10, step_size=30):
        """
        使用滑动窗口创建子序列，并根据给定步长处理序列。
        该方法返回 [T, FEATURE_DIM] 的 latent 序列。
        """
        T = X_seq.shape[0]

        # 确保 seq_len 小于等于 T
        assert seq_len <= T, "seq_len must be less than or equal to the number of timesteps in the sequence."

        # 使用滑动窗口生成所有子序列
        # 生成一个形状为 (T-seq_len+1, seq_len, F) 的子序列数组
        all_windows = np.lib.stride_tricks.sliding_window_view(X_seq, window_shape=(seq_len, X_seq.shape[1]))

        # 步长设置为 `step_size`
        # 这里选择从所有窗口中按步长选取子序列
        all_windows = all_windows[::step_size]

        # 形状变为 (num_windows, seq_len, F)，我们需要将其扩展为批次维度
        all_windows = np.expand_dims(all_windows, axis=0)  # (1, num_windows, seq_len, F)

        # 使用 encoder 一次性处理所有的子序列
        latents = encoder(all_windows.astype(np.float32)).numpy()  # (num_windows, FEATURE_DIM)

        # 创建一个零矩阵，用来存储最终的 latents 序列
        latents_full = np.zeros((T, latents.shape[-1]), dtype=np.float32)

        # 填充 latent 序列
        for t in range(len(latents)):
            latents_full[t * step_size: t * step_size + seq_len] = latents[t]

        return latents_full

    def rollout_meta_sarsa(self,X_train,X_val,encoder,seq_len,fea_dim, steps=30, epsilon=0.1):
        """
        X_seq: [T, F] 连续特征序列 (temp, humid, light, ac, heater, dehum, hum)
        labels: 可选，用于计算 reward
        """
        # FIX: 不使用整体 embedding；按时间步编码
        s_latent_seq = self.encode_timestep(encoder, X_train,seq_len=seq_len)  # [T, FEATURE_DIM]

        rewards_list = []
        actions_list = []

        for t in range(steps):
            # Q值近似：用当前 latent 的 Q 值 + 小噪声
            q_vals = self.q_net.predict(s_latent_seq[t:t+1], verbose=0)[0]
            q_vals = q_vals + np.random.randn(NUM_ACTIONS) * 0.01  # exploration noise

            # ε-greedy
            if np.random.rand() < epsilon:
                a = np.random.randint(NUM_ACTIONS)
            else:
                a = int(np.argmax(q_vals))

            bits = np.array([(a >> i) & 1 for i in range(NUM_SWITCH)], dtype=np.float32)
            actions_list.append(bits)

            # reward 计算
            if X_val is not None:
                r = self.compute_reward_batch(
                    np.array([X_val[min(t, len(X_val) - 1)]]),
                    np.array([bits]),
                )[0]
            else:
                r = 0.0
            rewards_list.append(r)

            # 更新状态：把动作 bits 写回 HVAC 列（影响下一步原始特征）
            if t + 1 < X_train.shape[0]:
                X_train[t + 1, 3:3 + NUM_SWITCH] = bits
                # 同步下一步 latent
                s_latent_seq[t + 1] = self.encode_per_timestep(encoder, X_train[t + 1:t + 2],seq_len,fea_dim)[0]
        print("Actions:", actions_list)
        print("Rewards:", rewards_list)


    # ------------------ 定义 Q 网络 ------------------
    def build_q_network(self,latent_dim=ENCODER_LATENT_DIM, num_actions=NUM_ACTIONS,dtype=tf.float32):
        inp = layers.Input(shape=(latent_dim,),dtype=tf.float32)
        x = inp
        for h in Q_HIDDEN:
            x = layers.Dense(h, activation="relu",dtype=tf.float32)(x)
        out = layers.Dense(num_actions, activation="linear",dtype=tf.float32)(x)
        return models.Model(inp, out)

    # ------------------ Helper ------------------
    def action_int_to_bits(self,a_int):
        return np.array([(a_int >> i) & 1 for i in range(NUM_SWITCH)], dtype=np.float32)


    def select_action_batch(self,q_values, eps):
        batch_size = q_values.shape[0]
        actions = np.zeros(batch_size, dtype=np.int32)
        for i in range(batch_size):
            if np.random.rand() < eps:
                actions[i] = np.random.randint(q_values.shape[1])
            else:
                actions[i] = int(np.argmax(q_values[i]))
        return actions


    def compute_reward_batch(self,labels, actions_bits):
        rewards = np.where(labels == 0, 1.0, np.where(labels == 1, -1.0, -2.0))
        rewards -= ACTION_COST * np.sum(actions_bits, axis=1)
        return rewards.astype(np.float32)

    def ensure_index_vector(self,x):
        """
        將索引向量保證為 1D tensor [batch_size]
        """
        x = tf.convert_to_tensor(x, dtype=tf.int32)
        x = tf.reshape(x, [-1])  # 保證 1D
        return x
    def ensure_vector(self, x, dtype=tf.float32):
        x = tf.convert_to_tensor(x, dtype=dtype)
        if len(x.shape) == 1:
            x = tf.expand_dims(x, axis=0)
        return x
    # ------------------ tf.function SARSA 更新 ------------------
    @tf.function
    def sarsa_batch_update(self, s_latent, a, r, s_next_latent, a_next, gamma=0.99, alpha=0.1):
        s_latent = self.ensure_vector(s_latent, dtype=tf.float32)  # 狀態向量
        s_next_latent = self.ensure_vector(s_next_latent, dtype=tf.float32)
        r = self.ensure_vector(r, dtype=tf.float32)  # reward
        a = self.ensure_index_vector(x=a)
        a_next = self.ensure_index_vector(x=a_next)
        q_values = self.q_net(s_latent, training=True)
        q_next_values = self.q_net(s_next_latent, training=False)
        batch_size = tf.shape(s_latent)[0]
        # 確保索引是 1D
        a = tf.reshape(a, [-1])  # shape [batch_size]
        a_next = tf.reshape(a_next, [-1])  # shape [batch_size]

        # indices shape [batch_size, 2]
        indices = tf.stack([tf.range(batch_size), a], axis=1)  # shape [batch_size, 2]

        q_s_a = tf.gather_nd(q_values, indices)  # shape [batch_size]
        q_s_next_a_next = tf.gather_nd(q_next_values, tf.stack([tf.range(batch_size), a_next], axis=1))
        q_s_next_a_next = tf.cast(q_s_next_a_next, tf.float32)
        r = tf.cast(r, tf.float32)
        # TD 目標
        target_updates = r + gamma * q_s_next_a_next  # shape [batch_size]
        # SARSA 更新：確保 1D
        updates = q_s_a + alpha * (target_updates - q_s_a)  # shape [batch_size]
        # TensorScatter 更新
        updates = tf.reshape(updates, [-1])  # [batch_size]
        target_q = tf.tensor_scatter_nd_update(q_values, indices, updates)
        with tf.GradientTape() as tape:
            pred_q = self.q_net(s_latent, training=True)
            loss = self.mse_loss(target_q, pred_q)
        grads = tape.gradient(loss, self.q_net.trainable_variables)
        self.q_optimizer.apply_gradients(zip(grads, self.q_net.trainable_variables))
        return loss


# ------------------ 测试 rollout ------------------
def test_rollout(qmodel,encoder,files, steps=30):
        df = pd.read_csv(np.random.choice(files))

        # 起始状态（第一行 -> (1,1,F)）
        next_row = df.iloc[0][["temp", "humid", "light", "ac", "heater", "dehum", "hum"]].values.astype(np.float32)
        next_row[0] /= 40.0
        next_row[1] /= 100.0
        next_row[2] /= 1000.0


        print("\nTest rollout:")
        for t in range(steps):

            s_latent = encoder(next_row[np.newaxis, np.newaxis, :]).numpy()[0]  # (FEATURE_DIM,)

            qv = qmodel.q_net.predict(s_latent.reshape(1, -1), verbose=0)[0]
            a = int(np.argmax(qv))
            bits = qmodel.action_int_to_bits(a)

            next_idx = min(t + 1, len(df) - 1)
            next_row = df.iloc[next_idx][["temp", "humid", "light", "ac", "heater", "dehum", "hum"]].values.astype(np.float32)
            next_row[0] /= 40.0
            next_row[1] /= 100.0
            next_row[2] /= 1000.0
            next_row[3:3 + NUM_SWITCH] = bits

            # FIX: 用 next_row 计算 next_latent（而不是误用 s_raw）
            #next_lat =  encoder(next_row[np.newaxis, np.newaxis, :]).numpy()[0]

            label_next = int(df.iloc[next_idx]["label"])
            r = qmodel.compute_reward_batch(np.array([label_next]), np.array([bits]))[0]
            print(f"t={t:02d} action={a:02d} bits={bits.tolist()} reward={r:.3f} label_next={label_next}")
            #s_latent = next_lat


def save_sarsa_tflite(q_net):
    # 在训练服务器上转换模型

    # 转换模型为TensorFlow Lite格式
    converter = tf.lite.TFLiteConverter.from_keras_model(q_net)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]  # 优化模型大小
    #converter.target_spec.supported_types = [tf.float16]  # 使用16位浮点数减小模型

    tflite_model = converter.convert()

    # 保存模型
    with open('hvac_controller.tflite', 'wb') as f:
        f.write(tflite_model)
    q_net.save("hvac_controller.h5")
    print(f"Model size: {len(tflite_model)} bytes")
# ------------------ 主程序 ------------------
'''
def main():
    # ------------------ 混合精度 ------------------
    from tensorflow.keras import mixed_precision
    policy = mixed_precision.Policy('float32')  # mixed_float16
    mixed_precision.set_global_policy(policy)

    eps = EPS_START
    files = [os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) if f.endswith(".csv")]

    meta_model = keras.models.load_model("sarsa/meta_model.h5")
    # ===== Meta model =====
    # meta_model = TFLiteModelWrapper("tflite_model_path")
    # Inspect model
    meta_model.summary()
    hvac_dense_layer = meta_model.get_layer("hvac_dense")  # lstm_encoder classifier
    encoder = keras.models.Model(inputs=meta_model.input, outputs=hvac_dense_layer.output)

    qmodel=QModel()
    qmodel.sarsa_full_batch_robust(encoder, SEQ_LEN, NUM_FEATURES)
    save_sarsa_tflite(qmodel.q_net)

    #test_rollout(encoder)



    df_all = load_all_csvs(DATA_DIR)
    X = df_all[["temp", "humid", "light", "ac", "heater", "dehum", "hum"]].values.astype(np.float32)
    y = df_all["label"].values.astype(np.int32)
    X[:, 0] /= 40.0
    X[:, 1] /= 100.0
    X[:, 2] /= 1000.0
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)
    qmodel.rollout_meta_sarsa(X_train, X_val, encoder, SEQ_LEN, NUM_FEATURES, steps=30, epsilon=0.1)

#if __name__ == "__main__":
#    main()
'''