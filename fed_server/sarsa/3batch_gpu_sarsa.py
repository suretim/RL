# -----------------------------
# 3sarsa_controller_full.py
# -----------------------------
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from sklearn.model_selection import train_test_split
from utils_module import generate_plant_sequence

from global_hyparm import *
# ------------------ 配置 ------------------


ENCODER_LATENT_DIM = 16
Q_HIDDEN = [64,64]

NUM_ACTIONS = 2 ^ NUM_SWITCH

LR_Q = 1e-3
SARSA_EPISODES = 200
BATCH_MAX = 128
GAMMA = 0.95
EPS_START = 0.3
EPS_END = 0.1
EPS_DECAY = 0.995
ACTION_COST = 0.05

# ------------------ 混合精度 ------------------
from tensorflow.keras import mixed_precision
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

eps = EPS_START

# ------------------ 数据生成 ------------------
os.makedirs(SAVE_DIR, exist_ok=True)
for i in range(NUM_FILES):
    file_path = os.path.join(SAVE_DIR, f"plant_seq_with_hvac_fail_{i}.csv")
    #print(file_path)
    if not os.path.exists(file_path):
        df = generate_plant_sequence(SAVE_DIR,seq_len=SEQ_LEN, noise_std=NOISE_STD)
        df.to_csv(file_path, index=False)

files = [os.path.join(SAVE_DIR,f) for f in os.listdir(SAVE_DIR) if f.endswith(".csv")]

# ------------------ 载入 CSV 预训练 encoder ------------------
def load_all_csvs(data_dir):
    dfs = [pd.read_csv(os.path.join(data_dir,f)) for f in os.listdir(data_dir) if f.endswith(".csv")]
    return pd.concat(dfs, axis=0).reset_index(drop=True)

df_all = load_all_csvs(SAVE_DIR)
X = df_all[["temp","humid","light","ac","heater","dehum","hum"]].values.astype(np.float32)
y = df_all["label"].values.astype(np.int32)
X[:,0]/=40.0; X[:,1]/=100.0; X[:,2]/=1000.0
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)

def build_classifier(input_dim=7, latent_dim=ENCODER_LATENT_DIM):
    inp = layers.Input(shape=(input_dim,))
    x = layers.Dense(64, activation="relu")(inp)
    x = layers.Dense(64, activation="relu")(x)
    latent = layers.Dense(latent_dim, activation="relu", name="latent")(x)
    out = layers.Dense(3, activation="softmax")(latent)
    return models.Model(inp, out)

clf = build_classifier()
clf.compile(optimizer=optimizers.Adam(1e-3), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
clf.fit(X_train, y_train, validation_data=(X_val,y_val), epochs=8, batch_size=512, verbose=2)

encoder = models.Model(clf.input, clf.get_layer("latent").output)

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
    return np.array([(a_int>>i)&1 for i in range(NUM_SWITCH)], dtype=np.float32)

def select_action_batch(q_values, eps):
    batch_size = q_values.shape[0]
    actions = np.zeros(batch_size,dtype=np.int32)
    for i in range(batch_size):
        if np.random.rand() < eps:
            actions[i] = np.random.randint(q_values.shape[1])
        else:
            actions[i] = int(np.argmax(q_values[i]))
    return actions

def compute_reward_batch0(labels, actions_bits):
    rewards = np.where(labels==0,1.0,np.where(labels==1,-1.0,-2.0))
    rewards -= ACTION_COST*np.sum(actions_bits, axis=1)
    return rewards.astype(np.float32)


def compute_reward_batch(labels, actions_bits, healthy_state=0,
                         energy_penalty=0.1, match_bonus=0.5):
    """
    labels: shape [B, num_feats]   e.g. [health, ac, dehum]
            health=0(健康)/1(不健康)，ac/dehum=0/1 表示推荐状态
    actions_bits: shape [B, 2]  每个位置是0/1，顺序为 [ac, dehum]
    healthy_state: 健康的label编号 (默认0)
    energy_penalty: 每开一个设备的能耗惩罚
    match_bonus: 动作与label一致时的奖励
    """
    # 提取健康状态
    health_state = labels[:, 0]
    hvac_targets = labels[:, 1:3]  # 只取 [ac, dehum]

    # 健康得分
    health_score = np.where(health_state == healthy_state, 1.0, -1.0)

    # 能耗惩罚
    energy_cost = energy_penalty * np.sum(actions_bits, axis=1)

    # AC 和 Dehum 匹配奖励
    match_score = np.sum((actions_bits == hvac_targets) & (hvac_targets == 1), axis=1) * match_bonus

    # 总奖励
    rewards = health_score - energy_cost + match_score
    return rewards

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
        loss = mse_loss(tf.expand_dims(target,1), tf.expand_dims(q_pred,1))

    grads = tape.gradient(loss, q_net.trainable_variables)
    q_optimizer.apply_gradients(zip(grads, q_net.trainable_variables))
    return loss


# ------------------ 完全批量化 SARSA 训练 ------------------
def sarsa_full_batch_robust():
    global eps

    for ep in range(1, SARSA_EPISODES+1):
        df = pd.read_csv(np.random.choice(files))
        T = len(df)
        total_reward = 0.0

        for start_idx in range(0, T-1, BATCH_MAX):
            end_idx = min(start_idx+BATCH_MAX+1, T)
            raw_seq = df.iloc[start_idx:end_idx][["temp","humid","light","ac","heater","dehum","hum"]].values.astype(np.float32)
            raw_seq[:,0]/=40.0; raw_seq[:,1]/=100.0; raw_seq[:,2]/=1000.0
            labels = df.iloc[start_idx:end_idx]["label"].values.astype(np.int32)
            seq_len = raw_seq.shape[0]
            if seq_len < 2:
                continue

            s_latent_seq = tf.convert_to_tensor(encoder.predict(raw_seq, batch_size=seq_len, verbose=0), dtype=tf.float16)

            q_vals_seq = q_net.predict(s_latent_seq[:-1], verbose=0)+ np.random.randn(NUM_ACTIONS)



            actions = select_action_batch(q_vals_seq, eps)
            actions_bits = np.array([action_int_to_bits(a) for a in actions])

            raw_next_seq = raw_seq[1:].copy()
            raw_next_seq[:,3:7] = actions_bits
            s_next_latent_seq = tf.convert_to_tensor(encoder.predict(raw_next_seq, batch_size=seq_len-1, verbose=0), dtype=tf.float16)

            rewards = compute_reward_batch(labels[1:], actions_bits)
            total_reward += np.sum(rewards)

            sarsa_batch_update(s_latent_seq[:-1],
                               tf.convert_to_tensor(actions, tf.int32),
                               s_next_latent_seq,
                               tf.convert_to_tensor(rewards, tf.float32))

        eps = max(EPS_END, eps*EPS_DECAY)
        if ep%10==0 or ep==1:
            print(f"Episode {ep}/{SARSA_EPISODES}  eps={eps:.3f}  total_reward={total_reward:.3f}")

# ------------------ 测试 rollout ------------------
def test_rollout(steps=30):
    df = pd.read_csv(np.random.choice(files))
    s_raw = df.iloc[0][["temp","humid","light","ac","heater","dehum","hum"]].values.astype(np.float32)
    s_raw[0]/=40.0; s_raw[1]/=100.0; s_raw[2]/=1000.0
    s_latent = encoder.predict(s_raw.reshape(1,-1), verbose=0)[0]
    print("\nTest rollout:")
    for t in range(steps):
        qv = q_net.predict(s_latent.reshape(1,-1), verbose=0)[0]
        a = int(np.argmax(qv))
        bits = action_int_to_bits(a)
        next_row = df.iloc[min(t+1,len(df)-1)][["temp","humid","light","ac","heater","dehum","hum"]].values.astype(np.float32)
        next_row[0]/=40.0; next_row[1]/=100.0; next_row[2]/=1000.0
        next_row[3:7] = bits
        next_lat = encoder.predict(next_row.reshape(1,-1), verbose=0)[0]
        label_next = int(df.iloc[min(t+1,len(df)-1)]["label"])
        r = compute_reward_batch(np.array([label_next]), np.array([bits]))[0]
        print(f"t={t:02d} action={a:02d} bits={bits.tolist()} reward={r:.3f} label_next={label_next}")
        s_latent = next_lat

# ------------------ 主程序 ------------------
if __name__ == "__main__":
    sarsa_full_batch_robust()
    test_rollout()
