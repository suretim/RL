# -----------------------------
# 3sarsa_controller_full_fixed.py
# -----------------------------
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from sklearn.model_selection import train_test_split
from utils_module import generate_plant_sequence
from utils_module import sample_tasks,load_csv_data

from global_hyparm import *
# ------------------ 配置 ------------------

ENCODER_LATENT_DIM = 16
Q_HIDDEN = [64, 64]

# FIX: 使用幂而不是按位异或；NUM_ACTIONS 由 NUM_SWITCH 决定
NUM_ACTIONS = 2 ** NUM_SWITCH  # e.g., NUM_SWITCH=4 -> 16

LR_Q = 1e-3
SARSA_EPISODES = 200
BATCH_MAX = 128
GAMMA = 0.95
EPS_START = 0.3
EPS_END = 0.1
EPS_DECAY = 0.995
ACTION_COST = 0.05

from utils_fisher import *


def encode_sequence(encoder, X_seq):
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


def encode_per_timestep(meta_model, X_seq):
    """
    将 [T, F] 的连续特征序列逐时间步送入 encoder（使用长度为1的序列），
    返回 [T, FEATURE_DIM] 的 latent 序列。
    这样避免修改现有的 LSTM encoder（其最终输出为全局 embedding）。
    """
    T = X_seq.shape[0]
    latents = []
    for t in range(T):
        x = X_seq[t:t+1][np.newaxis, :, :]  # (1, 1, F)
        z = meta_model.encoder(x).numpy()[0]  # (FEATURE_DIM,)
        latents.append(z)
    return np.stack(latents, axis=0)  # (T, FEATURE_DIM)






def rollout_meta_sarsa(meta_model, X_seq, labels=None, steps=30, epsilon=0.1):
    """
    X_seq: [T, F] 连续特征序列 (temp, humid, light, ac, heater, dehum, hum)
    labels: 可选，用于计算 reward
    """
    # FIX: 不使用整体 embedding；按时间步编码
    s_latent_seq = encode_per_timestep(meta_model, X_seq)  # [T, FEATURE_DIM]

    rewards_list = []
    actions_list = []

    for t in range(steps):
        # Q值近似：用当前 latent 的 Q 值 + 小噪声
        q_vals = q_net.predict(s_latent_seq[t:t+1], verbose=0)[0]
        q_vals = q_vals + np.random.randn(NUM_ACTIONS) * 0.01  # exploration noise

        # ε-greedy
        if np.random.rand() < epsilon:
            a = np.random.randint(NUM_ACTIONS)
        else:
            a = int(np.argmax(q_vals))

        bits = np.array([(a >> i) & 1 for i in range(NUM_SWITCH)], dtype=np.float32)
        actions_list.append(bits)

        # reward 计算
        if labels is not None:
            r = compute_reward_batch(
                np.array([labels[min(t, len(labels) - 1)]]),
                np.array([bits]),
            )[0]
        else:
            r = 0.0
        rewards_list.append(r)

        # 更新状态：把动作 bits 写回 HVAC 列（影响下一步原始特征）
        if t + 1 < X_seq.shape[0]:
            X_seq[t + 1, 3:3 + NUM_SWITCH] = bits
            # 同步下一步 latent
            s_latent_seq[t + 1] = encode_per_timestep(meta_model, X_seq[t + 1:t + 2])[0]
    print("Actions:", actions_list)
    print("Rewards:", rewards_list)


# ------ Service (orchestrates the pipeline) ------

def serv_pipline(num_classes=NUM_CLASSES, seq_len=SEQ_LEN, num_feats=NUM_FEATURES, feature_dim=FEATURE_DIM):
    load_glob = os.path.join(DATA_DIR, f"*.csv")

    # 1. 初始化 MetaModel
    model = MetaModel(num_classes=NUM_CLASSES, seq_len=SEQ_LEN, num_feats=NUM_FEATURES, feature_dim=FEATURE_DIM)

    # ===== Load data =====
    X_unlabeled, X_labeled, y_labeled, num_feats = load_csv_data(load_glob, seq_len)
    NUM_FEATS = num_feats  # sync global for model input shapes

    # ===== Build encoder =====
    lstm_encoder = model.build_lstm_encoder()

    # ===== Contrastive pretraining =====
    contrastive_opt = tf.keras.optimizers.Adam()
    ntxent = NTXentLoss(temperature=0.2)
    anchors, positives = model.make_contrastive_pairs(X_unlabeled)
    contrast_ds = tf.data.Dataset.from_tensor_slices((anchors, positives)).shuffle(2048).batch(BATCH_SIZE)
    for ep in range(EPOCHS_CONTRASTIVE):
        for a, p in contrast_ds:
            with tf.GradientTape() as tape:
                za = lstm_encoder(a, training=True)
                zp = lstm_encoder(p, training=True)
                loss = ntxent(za, zp)
            grads = tape.gradient(loss, lstm_encoder.trainable_variables)
            grads = [g if g is not None else tf.zeros_like(v) for g, v in zip(grads, lstm_encoder.trainable_variables)]
            contrastive_opt.apply_gradients(zip(grads, lstm_encoder.trainable_variables))
        print(f"[Contrastive] Epoch {ep + 1}/{EPOCHS_CONTRASTIVE}, loss={float(loss.numpy()):.4f}")

    # ===== Meta model =====
    meta_model = model.build_meta_model(lstm_encoder)
    meta_optimizer = tf.keras.optimizers.Adam(META_LR)
    model.set_trainable_layers(lstm_encoder, meta_model, ENCODER_MODE, LAST_N)

    memory = ReplayBuffer(capacity=REPLAY_CAPACITY)
    prev_weights = None
    fisher_matrix = None

    if X_labeled.size > 0:
        fisher_matrix = model.compute_fisher_matrix(meta_model, X_labeled, y_labeled)
        for ep in range(EPOCHS_META):
            tasks = sample_tasks(X_labeled, y_labeled, num_tasks=5)
            loss, acc, prev_weights = model.outer_update_with_lll(
                memory=memory,
                meta_model=meta_model,
                meta_optimizer=meta_optimizer,
                tasks=tasks,
                lr_inner=INNER_LR,
                prev_weights=prev_weights,
                fisher_matrix=fisher_matrix,
            )
            print(f"[Meta ] Epoch {ep + 1}/{EPOCHS_META}, loss={loss:.4f}, acc={acc:.4f}")
    else:
        print("Skip meta-learning: no labeled data.")

    # ===== Save assets =====
    if fisher_matrix is not None:
        model.save_fisher_and_weights(model=meta_model, fisher_matrix=fisher_matrix)

    # quick forward test
    dummy_x = np.random.rand(1, seq_len, num_feats).astype(np.float32)
    dummy_y = lstm_encoder(dummy_x)
    print("✅ Forward test shape:", dummy_y.shape)

    # 5. 导出 TFLite（保留原逻辑）
    try:
        model.save_tflite(meta_model.model, "meta_model.tflite")
    except Exception as e:
        print("[Warn] save_tflite failed:", e)

    #actions_list, rewards_list = rollout_meta_sarsa(meta_model, X_labeled.copy(), labels=y_labeled, steps=30, epsilon=0.1)
    #sarsa_full_batch_robust(meta_model)
    #test_rollout(meta_model)



# ------------------ 混合精度 ------------------
from tensorflow.keras import mixed_precision
policy = mixed_precision.Policy('float32') #mixed_float16
mixed_precision.set_global_policy(policy)

eps = EPS_START

# ------------------ 数据生成 ------------------
os.makedirs(DATA_DIR, exist_ok=True)
for i in range(NUM_FILES):
    file_path = os.path.join(DATA_DIR, f"plant_seq_with_hvac_fail_{i}.csv")
    if not os.path.exists(file_path):
        df = generate_plant_sequence(DATA_DIR, seq_len=SEQ_LEN, noise_std=NOISE_STD)
        df.to_csv(file_path, index=False)

files = [os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) if f.endswith(".csv")]

# ------------------ 载入 CSV 预训练 encoder ------------------
def load_all_csvs(data_dir):
    dfs = [pd.read_csv(os.path.join(data_dir, f)) for f in os.listdir(data_dir) if f.endswith(".csv")]
    return pd.concat(dfs, axis=0).reset_index(drop=True)

df_all = load_all_csvs(DATA_DIR)
X = df_all[["temp", "humid", "light", "ac", "heater", "dehum", "hum"]].values.astype(np.float32)
y = df_all["label"].values.astype(np.int32)
X[:, 0] /= 40.0
X[:, 1] /= 100.0
X[:, 2] /= 1000.0
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)

# ------------------ 定义 Q 网络 ------------------
def build_q_network(latent_dim=ENCODER_LATENT_DIM, num_actions=NUM_ACTIONS):
    inp = layers.Input(shape=(latent_dim,))
    x = inp
    for h in Q_HIDDEN:
        x = layers.Dense(h, activation="relu")(x)
    out = layers.Dense(num_actions, activation="linear")(x)
    return models.Model(inp, out)

q_net = build_q_network()
q_optimizer = optimizers.Adam(LR_Q)
mse_loss = tf.keras.losses.MeanSquaredError()

# ------------------ Helper ------------------
def action_int_to_bits(a_int):
    return np.array([(a_int >> i) & 1 for i in range(NUM_SWITCH)], dtype=np.float32)


def select_action_batch(q_values, eps):
    batch_size = q_values.shape[0]
    actions = np.zeros(batch_size, dtype=np.int32)
    for i in range(batch_size):
        if np.random.rand() < eps:
            actions[i] = np.random.randint(q_values.shape[1])
        else:
            actions[i] = int(np.argmax(q_values[i]))
    return actions


def compute_reward_batch(labels, actions_bits):
    rewards = np.where(labels == 0, 1.0, np.where(labels == 1, -1.0, -2.0))
    rewards -= ACTION_COST * np.sum(actions_bits, axis=1)
    return rewards.astype(np.float32)


# ------------------ tf.function SARSA 更新 ------------------
@tf.function
def sarsa_batch_update(s_latent, actions, s_next_latent, rewards):
    batch_idx = tf.range(tf.shape(actions)[0], dtype=tf.int32)
    with tf.GradientTape() as tape:
        q_pred_all = q_net(s_latent, training=True)  # shape (batch, num_actions)
        q_pred = tf.gather_nd(q_pred_all, tf.stack([batch_idx, actions], axis=1))

        q_next_all = q_net(s_next_latent, training=False)
        next_actions = tf.argmax(q_next_all, axis=1, output_type=tf.int32)
        q_next = tf.gather_nd(q_next_all, tf.stack([batch_idx, next_actions], axis=1))

        # 转成同一 dtype
        rewards = tf.cast(rewards, q_next.dtype)

        target = rewards + GAMMA * q_next
        loss = mse_loss(tf.expand_dims(target, 1), tf.expand_dims(q_pred, 1))

    grads = tape.gradient(loss, q_net.trainable_variables)
    q_optimizer.apply_gradients(zip(grads, q_net.trainable_variables))
    return loss


# ------------------ 完全批量化 SARSA 训练 ------------------
def sarsa_full_batch_robust(meta_model):
    global eps

    for ep in range(1, SARSA_EPISODES + 1):
        df = pd.read_csv(np.random.choice(files))
        T = len(df)
        total_reward = 0.0

        for start_idx in range(0, T - 1, BATCH_MAX):
            end_idx = min(start_idx + BATCH_MAX + 1, T)
            raw_seq = df.iloc[start_idx:end_idx][["temp", "humid", "light", "ac", "heater", "dehum", "hum"]].values.astype(np.float32)
            raw_seq[:, 0] /= 40.0
            raw_seq[:, 1] /= 100.0
            raw_seq[:, 2] /= 1000.0
            labels = df.iloc[start_idx:end_idx]["label"].values.astype(np.int32)
            seq_len = raw_seq.shape[0]
            if seq_len < 2:
                continue

            # FIX: 按时间步编码，得到 [seq_len, FEATURE_DIM]
            s_latent_seq = encode_per_timestep(meta_model, raw_seq)

            # 预测 Q 值 + 噪声（匹配形状）
            q_pred = q_net.predict(s_latent_seq[:-1], verbose=0)  # (seq_len-1, NUM_ACTIONS)
            noise = np.random.randn(*q_pred.shape) * 0.01  # FIX: 广播为同形状
            q_vals_seq = q_pred + noise

            actions = select_action_batch(q_vals_seq, eps)
            actions_bits = np.array([action_int_to_bits(a) for a in actions])

            raw_next_seq = raw_seq[1:].copy()
            raw_next_seq[:, 3:3 + NUM_SWITCH] = actions_bits
            s_next_latent_seq = encode_per_timestep(meta_model, raw_next_seq)

            rewards = compute_reward_batch(labels[1:], actions_bits)
            total_reward += np.sum(rewards)

            sarsa_batch_update(
                tf.convert_to_tensor(s_latent_seq[:-1], dtype=tf.float32),
                tf.convert_to_tensor(actions, tf.int32),
                tf.convert_to_tensor(s_next_latent_seq, dtype=tf.float32),
                tf.convert_to_tensor(rewards, tf.float32),
            )

        eps = max(EPS_END, eps * EPS_DECAY)
        if ep % 10 == 0 or ep == 1:
            print(f"Episode {ep}/{SARSA_EPISODES}  eps={eps:.3f}  total_reward={total_reward:.3f}")


# ------------------ 测试 rollout ------------------
def test_rollout(meta_model, steps=30):
    df = pd.read_csv(np.random.choice(files))

    # 起始状态（第一行 -> (1,1,F)）
    first_row = df.iloc[0][["temp", "humid", "light", "ac", "heater", "dehum", "hum"]].values.astype(np.float32)
    first_row[0] /= 40.0
    first_row[1] /= 100.0
    first_row[2] /= 1000.0

    s_latent = meta_model.encoder(first_row[np.newaxis, np.newaxis, :]).numpy()[0]  # (FEATURE_DIM,)

    print("\nTest rollout:")
    for t in range(steps):
        qv = q_net.predict(s_latent.reshape(1, -1), verbose=0)[0]
        a = int(np.argmax(qv))
        bits = action_int_to_bits(a)

        next_idx = min(t + 1, len(df) - 1)
        next_row = df.iloc[next_idx][["temp", "humid", "light", "ac", "heater", "dehum", "hum"]].values.astype(np.float32)
        next_row[0] /= 40.0
        next_row[1] /= 100.0
        next_row[2] /= 1000.0
        next_row[3:3 + NUM_SWITCH] = bits

        # FIX: 用 next_row 计算 next_latent（而不是误用 s_raw）
        next_lat = meta_model.encoder(next_row[np.newaxis, np.newaxis, :]).numpy()[0]

        label_next = int(df.iloc[next_idx]["label"])
        r = compute_reward_batch(np.array([label_next]), np.array([bits]))[0]
        print(f"t={t:02d} action={a:02d} bits={bits.tolist()} reward={r:.3f} label_next={label_next}")
        s_latent = next_lat


# ------------------ 主程序 ------------------
if __name__ == "__main__":
    serv_pipline(num_classes=NUM_CLASSES, seq_len=SEQ_LEN, num_feats=NUM_FEATURES, feature_dim=FEATURE_DIM)
