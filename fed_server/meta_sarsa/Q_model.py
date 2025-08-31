import pandas as pd
from tensorflow import keras
from tensorflow.keras import optimizers

from sklearn.model_selection import train_test_split

# 依赖：NUM_SWITCH, DATA_DIR, SEQ_LEN, NUM_FEATURES 等
from utils_fisher import *                 # 如果无需本文件的函数，可以移除这行

# ============== 配置 ==============
ENCODER_LATENT_DIM = 16
Q_HIDDEN = [64, 64]

# 注意：NUM_ACTIONS 必须由 NUM_SWITCH 推导
NUM_ACTIONS = 2 ** NUM_SWITCH

LR_Q = 1e-3
SARSA_EPISODES = 200
BATCH_MAX = 32
GAMMA = 0.95
EPS_START = 0.3
EPS_END = 0.1
EPS_DECAY = 0.995
ACTION_COST = 0.05

# ============== 混合精度（此处保持 float32 更稳） ==============
from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy('float32')

# 数据文件列表
files = [os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) if f.endswith(".csv")]

def normalize_row(row):
    row = row.astype(np.float32)
    row[0] /= 40.0     # temp
    row[1] /= 100.0    # humid
    row[2] /= 1000.0   # light
    return row

def make_windows(X, seq_len, step_size=1):
    """
    X: (T, F) -> windows: (N, seq_len, F), idx: list of start indices
    N = floor((T - seq_len)/step_size) + 1
    """
    T = X.shape[0]
    if T < seq_len:
        return np.empty((0, seq_len, X.shape[1]), dtype=np.float32), []
    starts = list(range(0, T - seq_len + 1, step_size))
    windows = np.stack([X[s:s+seq_len] for s in starts], axis=0).astype(np.float32)
    return windows, starts

class QModel:
    def __init__(self, latent_dim=ENCODER_LATENT_DIM, num_actions=NUM_ACTIONS):
        self.q_net = self.build_q_network(latent_dim, num_actions)
        self.optimizer = optimizers.Adam(LR_Q)
        self.eps = EPS_START

    # ---------- Q网络 ----------
    def build_q_network(self, latent_dim, num_actions):
        inp = layers.Input(shape=(latent_dim,), name="latent_in")
        x = inp
        for h in Q_HIDDEN:
            x = layers.Dense(h, activation="relu")(x)
        out = layers.Dense(num_actions, activation=None, name="q_out")(x)
        return models.Model(inp, out, name="q_net")

    # ---------- 工具 ----------
    @staticmethod
    def action_int_to_bits(a_int):
        return np.array([(a_int >> i) & 1 for i in range(NUM_SWITCH)], dtype=np.float32)

    @staticmethod
    def select_action_batch(q_values, eps):
        # q_values: (B, A)
        if q_values.shape[0] == 0:
            return np.zeros((0,), dtype=np.int32)
        greedy = np.argmax(q_values, axis=1).astype(np.int32)
        rand_mask = (np.random.rand(q_values.shape[0]) < eps)
        random_actions = np.random.randint(q_values.shape[1], size=q_values.shape[0])
        actions = np.where(rand_mask, random_actions, greedy).astype(np.int32)
        return actions

    @staticmethod
    def compute_reward_batch(labels, actions_bits):
        """
        labels: (B,) 值域示例 {0,1,其他}
        actions_bits: (B, NUM_SWITCH)
        """
        # 示例奖励：0 -> +1, 1 -> -1, else -> -2
        rewards = np.where(labels == 0, 1.0, np.where(labels == 1, -1.0, -2.0))
        rewards = rewards - ACTION_COST * np.sum(actions_bits, axis=1)
        return rewards.astype(np.float32)

    def encode_sequence(self, encoder, X_seq):
        """
        X_seq: (B, seq_len, F) 或 (1, seq_len, F)
        返回 (B, latent_dim)
        """
        return encoder(X_seq).numpy().astype(np.float32)

    def encode_per_timestep(self, encoder, raw, seq_len=None, step_size=1):
        """
        Args:
            encoder: 模型 (輸入 [B, seq_len, F] -> 輸出 latent)
            raw: numpy array, shape [T, F]
            seq_len: 每個窗口長度 (int 或 None)，None 時自動決定
            step_size: 滑動步長

        Returns:
            s_latent_seq: shape [N, D]，所有窗口 latent
            starts: 每個窗口對應的起始索引
        """
        T = raw.shape[0]

        # --- 動態決定窗口長度 ---
        if seq_len is None:
            # 例如取 T 的 1/10，至少 5
            seq_len = max(10, T // 20)

        # --- 滑動窗口切片 ---
        starts = list(range(0, T - seq_len + 1, step_size))
        seqs = [raw[i:i + seq_len] for i in starts]

        if not seqs:
            return np.zeros((0, encoder.output_shape[-1])), []

        seqs = np.array(seqs)  # [N, seq_len, F]

        # --- 編碼 ---
        s_latent_seq = encoder.predict(seqs, verbose=0)  # [N, D]

        return s_latent_seq, starts


    @tf.function
    def _sarsa_batch_update_tf(self, s_latent, actions, s_next_latent, rewards):
        # s_latent: (B, D), actions: (B,), s_next_latent: (B, D), rewards: (B,)
        with tf.GradientTape() as tape:
            q_pred_all = self.q_net(s_latent, training=True)                 # (B, A)
            batch_idx = tf.range(tf.shape(actions)[0], dtype=tf.int32)
            q_pred = tf.gather_nd(q_pred_all, tf.stack([batch_idx, actions], axis=1))  # (B,)

            # target 使用 next state's max-Q（SARSA可改：用 next action 的 Q；此处做 Double-check）
            q_next_all = self.q_net(s_next_latent, training=False)           # (B, A)
            next_actions = tf.argmax(q_next_all, axis=1, output_type=tf.int32)
            q_next = tf.gather_nd(q_next_all, tf.stack([batch_idx, next_actions], axis=1))  # (B,)

            rewards = tf.cast(rewards, q_next.dtype)
            target = rewards + GAMMA * q_next                                # (B,)

            loss = tf.reduce_mean(tf.square(target - q_pred))
        grads = tape.gradient(loss, self.q_net.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.q_net.trainable_variables))
        return loss

    def sarsa_full_batch_robust(self, encoder, seq_len, step_size=1):
        """
        每次从 CSV 采样一段，构造 (s_t, a_t, r_{t+1}, s_{t+1}) 的全批量样本并更新。
        """
        for ep in range(1, SARSA_EPISODES + 1):
            df = pd.read_csv(np.random.choice(files))

            raw = df[["temp", "humid", "light", "ac", "heater", "dehum", "hum"]].values.astype(np.float32)
            labels = df["label"].values.astype(np.int32)

            # 归一化非控制量
            raw[:, 0] /= 40.0
            raw[:, 1] /= 100.0
            raw[:, 2] /= 1000.0

            # 计算所有窗口 latent
            s_latent_seq, starts = self.encode_per_timestep(encoder, raw)  # (N, D)
            N = s_latent_seq.shape[0]
            if N <= 1:
                # 不足以形成 (s_t, s_{t+1})
                continue

            # 预测当前 Q(s_t, ·)
            q_pred = self.q_net.predict(s_latent_seq[:-1], verbose=0)  # (N-1, A)
            # 探索噪声（很小）
            q_pred = q_pred + np.random.randn(*q_pred.shape).astype(np.float32) * 0.01

            # ε-greedy 选行动
            actions = self.select_action_batch(q_pred, self.eps)              # (N-1,)
            actions_bits = np.stack([self.action_int_to_bits(a) for a in actions], axis=0)  # (N-1, NUM_SWITCH)

            # 奖励：使用下一步窗口的标签（与 s_{t+1} 对齐）
            label_next_idx = [min(s + seq_len, len(labels) - 1) for s in starts[:-1]]
            rewards = self.compute_reward_batch(labels=np.array(label_next_idx, dtype=np.int32)*0 + labels[label_next_idx],
                                                actions_bits=actions_bits)

            # 构造 s_next
            s_next_latent_seq = s_latent_seq[1:]                              # (N-1, D)

            # 分批更新，避免过大 batch
            total_loss = 0.0
            B = BATCH_MAX
            for i in range(0, len(actions), B):
                j = min(i + B, len(actions))
                if j - i <= 0: continue
                loss = self._sarsa_batch_update_tf(
                    tf.convert_to_tensor(s_latent_seq[i:j], dtype=tf.float32),
                    tf.convert_to_tensor(actions[i:j], dtype=tf.int32),
                    tf.convert_to_tensor(s_next_latent_seq[i:j], dtype=tf.float32),
                    tf.convert_to_tensor(rewards[i:j], dtype=tf.float32),
                )
                total_loss += float(loss.numpy())

            # 衰减 epsilon
            self.eps = max(EPS_END, self.eps * EPS_DECAY)

            if ep % 10 == 0 or ep == 1:
                print(f"Episode {ep:03d}/{SARSA_EPISODES} | eps={self.eps:.3f} | loss≈{total_loss:.4f} | samples={len(actions)}")

    def rollout_meta_sarsa(self, X_train, X_val, encoder, seq_len, steps=30, step_size=1, epsilon=0.1):
        """
        仅用于快速 sanity check：基于编码的 ε-贪婪 rollout。
        """
        latents, starts = self.encode_per_timestep(encoder, X_train, seq_len=seq_len, step_size=step_size)
        if latents.shape[0] == 0:
            print("No latent windows; check seq_len/step_size.")
            return

        rewards_list, actions_list = [], []
        for t in range(min(steps, latents.shape[0])):
            q_vals = self.q_net.predict(latents[t:t+1], verbose=0)[0]
            q_vals = q_vals + np.random.randn(NUM_ACTIONS).astype(np.float32) * 0.01
            if np.random.rand() < epsilon:
                a = np.random.randint(NUM_ACTIONS)
            else:
                a = int(np.argmax(q_vals))
            bits = self.action_int_to_bits(a)
            actions_list.append(bits)

            if X_val is not None and len(X_val) > 0:
                idx = min(starts[t] + seq_len, len(X_val) - 1)
                r = self.compute_reward_batch(np.array([int(X_val[idx, 0]*0)]),  # 这里没有真实标签，仅演示；建议换成 df 的 label
                                              np.array([bits]))[0]
            else:
                r = 0.0
            rewards_list.append(r)

        print("Actions:", [a.tolist() for a in actions_list])
        print("Rewards:", rewards_list)


# ============== 测试 rollout ==============
def test_rollout(qmodel, encoder, steps=30):
    df = pd.read_csv(np.random.choice(files))

    next_row = df.iloc[0][["temp", "humid", "light", "ac", "heater", "dehum", "hum"]].values.astype(np.float32)
    next_row = normalize_row(next_row)

    print("\nTest rollout:")
    for t in range(steps):
        # 假设 encoder 支持 (B, T=1, F)
        s_latent = encoder(next_row[np.newaxis, np.newaxis, :]).numpy()[0]  # (latent_dim,)

        qv = qmodel.q_net.predict(s_latent.reshape(1, -1), verbose=0)[0]
        a = int(np.argmax(qv))
        bits = qmodel.action_int_to_bits(a)

        next_idx = min(t + 1, len(df) - 1)
        next_row = df.iloc[next_idx][["temp", "humid", "light", "ac", "heater", "dehum", "hum"]].values.astype(np.float32)
        next_row = normalize_row(next_row)
        next_row[3:3 + NUM_SWITCH] = bits

        label_next = int(df.iloc[next_idx]["label"]) if "label" in df.columns else 0
        r = qmodel.compute_reward_batch(np.array([label_next]), np.array([bits]))[0]
        print(f"t={t:02d} action={a:02d} bits={bits.tolist()} reward={r:.3f} label_next={label_next}")


# ============== 数据工具 ==============
def load_all_csvs(data_dir):
    dfs = [pd.read_csv(os.path.join(data_dir, f)) for f in os.listdir(data_dir) if f.endswith(".csv")]
    return pd.concat(dfs, axis=0).reset_index(drop=True)

def save_sarsa_tflite(q_net):
    converter = tf.lite.TFLiteConverter.from_keras_model(q_net)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    with open('../hvac_controller.tflite', 'wb') as f:
        f.write(tflite_model)
    q_net.save("hvac_controller.h5")
    print(f"Model size: {len(tflite_model)} bytes")


# ============== 主程序 ==============
def main():
    # 载入 meta encoder：取名为 "hvac_dense" 的那层的输出
    meta_model = keras.models.load_model("../sarsa/meta_model.h5")
    meta_model.summary()
    hvac_dense_layer = meta_model.get_layer("hvac_dense")
    #inputs = tf.keras.Input(shape=(None, 7))  # None = 任意長度序列
    #x = tf.keras.layers.LSTM(64)(inputs)  # LSTM 可以處理可變長度
    #encoder = tf.keras.Model(inputs, x)

    encoder = keras.models.Model(inputs=meta_model.input, outputs=hvac_dense_layer.output)

    qmodel = QModel(latent_dim=ENCODER_LATENT_DIM, num_actions=NUM_ACTIONS)
    # 训练
    qmodel.sarsa_full_batch_robust(encoder, seq_len=SEQ_LEN, step_size=1)
    save_sarsa_tflite(qmodel.q_net)

    # Sanity rollout（可选）
    df_all = load_all_csvs(DATA_DIR)
    X = df_all[["temp", "humid", "light", "ac", "heater", "dehum", "hum"]].values.astype(np.float32)
    y = df_all["label"].values.astype(np.int32) if "label" in df_all.columns else np.zeros((len(df_all),), dtype=np.int32)
    X[:, 0] /= 40.0
    X[:, 1] /= 100.0
    X[:, 2] /= 1000.0
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y if y.sum() else None)
    qmodel.rollout_meta_sarsa(X_train, X_val, encoder, seq_len=SEQ_LEN, steps=30, step_size=1, epsilon=0.1)

if __name__ == "__main__":
    main()
