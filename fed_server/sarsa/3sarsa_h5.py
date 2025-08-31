import tensorflow as tf
import numpy as np
# -----------------------------
# 3sarsa_controller_full_fixed.py
# -----------------------------
from tensorflow.keras import layers, models, optimizers
from sklearn.model_selection import train_test_split

from global_hyparm import *
from lstm.train_meta_s_flwr import q_net

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

from utils_fisher import *

# ------------------ 混合精度 ------------------
from tensorflow.keras import mixed_precision
policy = mixed_precision.Policy('float32') #mixed_float16
mixed_precision.set_global_policy(policy)

eps = EPS_START
files = [os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) if f.endswith(".csv")]

class QModel:
    def __init__(self):
        self.q_net = self.build_q_network()


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

    def encode_per_timestep(encoder, X_seq, seq_len=10, step_size=30):
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
        s_latent_seq = self.encode_per_timestep(encoder, X_train,seq_len,fea_dim)  # [T, FEATURE_DIM]

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
    def build_q_network(self,latent_dim=ENCODER_LATENT_DIM, num_actions=NUM_ACTIONS):
        inp = layers.Input(shape=(latent_dim,))
        x = inp
        for h in Q_HIDDEN:
            x = layers.Dense(h, activation="relu")(x)
        out = layers.Dense(num_actions, activation="linear")(x)
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


    # ------------------ tf.function SARSA 更新 ------------------
    @tf.function
    def sarsa_batch_update(self,s_latent, actions, s_next_latent, rewards):
        batch_idx = tf.range(tf.shape(actions)[0], dtype=tf.int32)
        # Before tf.stack, check if batch is empty
        if batch_idx.shape[0] == 0:
            print("Warning: Empty batch detected!")
            # Return early or handle empty case
            return None  # or appropriate return value

        with tf.GradientTape() as tape:
            q_pred_all = q_net(s_latent, training=True)  # shape (batch, num_actions)
            q_pred = tf.gather_nd(q_pred_all, tf.stack([batch_idx, actions], axis=1))

            q_next_all = q_net(s_next_latent, training=False)
            next_actions = tf.argmax(q_next_all, axis=1, output_type=tf.int32)


            if batch_idx.shape[0] != next_actions.shape[0]:
                print(f"Shape mismatch: batch_idx {batch_idx.shape}, next_actions {next_actions.shape}")
                # Use the smaller size
                min_size = min(batch_idx.shape[0], next_actions.shape[0])
                batch_idx = batch_idx[:min_size]
                next_actions = next_actions[:min_size]

            q_next = tf.gather_nd(q_next_all, tf.stack([batch_idx, next_actions], axis=1))


            #q_next = tf.gather_nd(q_next_all, tf.stack([batch_idx, next_actions], axis=1))

            # 转成同一 dtype
            rewards = tf.cast(rewards, q_next.dtype)

            target = rewards + GAMMA * q_next
            mse_loss = tf.keras.losses.MeanSquaredError()
            loss = mse_loss(tf.expand_dims(target, 1), tf.expand_dims(q_pred, 1))

        grads = tape.gradient(loss, q_net.trainable_variables)
        q_optimizer = optimizers.Adam(LR_Q)
        q_optimizer.apply_gradients(zip(grads, q_net.trainable_variables))
        return loss


    # ------------------ 完全批量化 SARSA 训练 ------------------
    def sarsa_full_batch_robust(self,encoder,seq_len,num_feats):
        global eps

        for ep in range(1, SARSA_EPISODES + 1):
            df = pd.read_csv(np.random.choice(files))
            T = len(df)
            total_reward = 0.0

            for start_idx in range(0, T - 1, BATCH_MAX):
                end_idx = min(start_idx + seq_len + 1, T)
                raw_seq = df.iloc[start_idx:end_idx][["temp", "humid", "light", "ac", "heater", "dehum", "hum"]].values.astype(np.float32)
                raw_seq[:, 0] /= 40.0
                raw_seq[:, 1] /= 100.0
                raw_seq[:, 2] /= 1000.0
                labels = df.iloc[start_idx:end_idx]["label"].values.astype(np.int32)
                batch_len = raw_seq.shape[0]
                if batch_len < seq_len:
                    continue

                # FIX: 按时间步编码，得到 [seq_len, FEATURE_DIM]
                s_latent_seq = self.encode_per_timestep(encoder, raw_seq,seq_len,num_feats)
                # Debug the input shape
                #print(f"Input shape: {s_latent_seq[:-1].shape}")
                #print(f"Expected input shape: {q_net.input_shape}")

                # Minimal fix - just ensure q_pred is always defined
                try:
                    q_pred = self.q_net.predict(s_latent_seq[:-1], verbose=0)
                except:
                    # Create dummy predictions with correct shape
                    q_pred = np.zeros((len(s_latent_seq) - 1, NUM_ACTIONS))

                #noise = np.random.randn(*q_pred.shape) * 0.01
                #q_pred  = q_pred + noise
                # 预测 Q 值 + 噪声（匹配形状）
                #q_pred = self.q_net.predict(s_latent_seq[:-1], verbose=0)  # (seq_len-1, NUM_ACTIONS)
                noise = np.random.randn(*q_pred.shape) * 0.01  # FIX: 广播为同形状
                q_vals_seq = q_pred + noise

                actions = self.select_action_batch(q_vals_seq, eps)
                actions_bits = np.array([self.action_int_to_bits(a) for a in actions])
                raw_next_seq = raw_seq[1:].copy()
                # Before assignment, check if actions_bits is empty
                if actions_bits.size == 0:
                    print("Warning: actions_bits is empty!")
                    # Handle empty case - either skip or use default values
                    actions_bits = np.zeros((raw_next_seq.shape[0], NUM_SWITCH))  # Fill with zeros
                    # OR skip the assignment entirely if appropriate
                    # continue

                raw_next_seq[:, 3:3 + NUM_SWITCH] = actions_bits

                #
                #raw_next_seq[:, 3:3 + NUM_SWITCH] = actions_bits
                s_next_latent_seq = self.encode_per_timestep(encoder, raw_next_seq,seq_len,num_feats)

                rewards = self.compute_reward_batch(labels[1:], actions_bits)
                total_reward += np.sum(rewards)

                self.sarsa_batch_update(
                    tf.convert_to_tensor(s_latent_seq[:-1], dtype=tf.float32),
                    tf.convert_to_tensor(actions, tf.int32),
                    tf.convert_to_tensor(s_next_latent_seq, dtype=tf.float32),
                    tf.convert_to_tensor(rewards, tf.float32),
                )

            eps = max(EPS_END, eps * EPS_DECAY)
            if ep % 10 == 0 or ep == 1:
                print(f"Episode {ep}/{SARSA_EPISODES}  eps={eps:.3f}  total_reward={total_reward:.3f}")

# ------------------ 测试 rollout ------------------
def test_rollout(qmodel,encoder, steps=30):
        df = pd.read_csv(np.random.choice(files))

        # 起始状态（第一行 -> (1,1,F)）
        next_row = df.iloc[0][["temp", "humid", "light", "ac", "heater", "dehum", "hum"]].values.astype(np.float32)
        next_row[0] /= 40.0
        next_row[1] /= 100.0
        next_row[2] /= 1000.0


        print("\nTest rollout:")
        for t in range(steps):

            s_latent = encoder(next_row[np.newaxis, np.newaxis, :]).numpy()[0]  # (FEATURE_DIM,)

            qv = q_net.predict(s_latent.reshape(1, -1), verbose=0)[0]
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

# ------------------ 载入 CSV 预训练 encoder ------------------
def load_all_csvs(data_dir):
    dfs = [pd.read_csv(os.path.join(data_dir, f)) for f in os.listdir(data_dir) if f.endswith(".csv")]
    return pd.concat(dfs, axis=0).reset_index(drop=True)

def save_sarsa_tflite(q_net):
    # 在训练服务器上转换模型

    # 转换模型为TensorFlow Lite格式
    converter = tf.lite.TFLiteConverter.from_keras_model(q_net)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]  # 优化模型大小
    #converter.target_spec.supported_types = [tf.float16]  # 使用16位浮点数减小模型

    tflite_model = converter.convert()

    # 保存模型
    with open('../hvac_controller.tflite', 'wb') as f:
        f.write(tflite_model)
    q_net.save("hvac_controller.h5")
    print(f"Model size: {len(tflite_model)} bytes")
# ------------------ 主程序 ------------------
def main():
    global NUM_CLASSES, SEQ_LEN, NUM_FEATURES, FEATURE_DIM
    meta_model = keras.models.load_model("meta_model.h5")
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

if __name__ == "__main__":
    main()
