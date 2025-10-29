import tensorflow as tf
import numpy as np
# -----------------------------
# 3sarsa_controller_full_fixed.py
# -----------------------------

from tensorflow.keras import layers, models, optimizers
from sklearn.model_selection import train_test_split
from tensorflow import keras

import pandas as pd
import argparse
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))  # RL 根目录
from global_hyparm import *
# ------------------ 配置 ------------------
from utils_module import generate_plant_sequence,load_all_csvs
#from utils_fisher import MetaModel
from utils_QModel import QModel
DATA_DIR  = "../../../data/sarsa"
META_OUT_TF="../meta_lstm_classifier.tflite"
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


ENCODER_MODE="freeze"
LAST_N =1

def rollout_meta_sarsa(meta_model, q_net, X_train, X_val=None, seq_len=10, steps=30, epsilon=0.1):
    """
    使用元学习的 Sarsa 算法模拟回合。
    """
    # 获取潜在特征序列，按时间步编码
    s_latent_seq = q_net.encode_per_timestep(meta_model, X_train, seq_len, steps)  # [T, FEATURE_DIM]

    rewards_list = []
    actions_list = []

    for t in range(min(steps, X_train.shape[0] - 1)):
        # 计算 Q 值并添加探索噪声
        q_vals = q_net.predict(s_latent_seq[t:t+1], verbose=0)[0]
        q_vals = q_vals + np.random.randn(NUM_ACTIONS) * 0.01  # exploration noise

        # 选择动作 (ε-greedy 策略)
        if np.random.rand() < epsilon:
            a = np.random.randint(NUM_ACTIONS)
        else:
            a = int(np.argmax(q_vals))

        # 将动作转换为 bit 数组
        bits = np.array([(a >> i) & 1 for i in range(NUM_SWITCH)], dtype=np.float32)
        actions_list.append(bits)

        # 计算奖励
        if X_val is not None:
            r = q_net.compute_reward_batch(
                np.array([X_val[min(t, len(X_val) - 1)]]),
                np.array([bits]),
            )[0]
        else:
            r = 0.0
        rewards_list.append(r)

        # 更新环境状态：动作影响 HVAC 列，修改 X_train
        if t + 1 < X_train.shape[0]:
            X_train[t + 1, 3:3 + NUM_SWITCH] = bits
            # 更新潜在特征
            s_latent_seq[t + 1] = q_net.encode_per_timestep(meta_model, X_train[t + 1:t + 2], seq_len)[0]

    print("Actions:", actions_list)
    print("Rewards:", rewards_list)

    return actions_list, rewards_list


# ------------------ tf.function SARSA 更新 ------------------
@tf.function
def sarsa_batch_update(q_net,s_latent, actions, s_next_latent, rewards):
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
        loss = q_net.mse_loss(tf.expand_dims(target, 1), tf.expand_dims(q_pred, 1))

    grads = tape.gradient(loss, q_net.trainable_variables)
    q_net.q_optimizer.apply_gradients(zip(grads, q_net.trainable_variables))
    return loss

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

# ------------------ 测试 rollout ------------------
def test_rollout(q_net,encoder,files, steps=30):
    df = pd.read_csv(np.random.choice(files))

    # 起始状态（第一行 -> (1,1,F)）
    first_row = df.iloc[0][["temp", "humid", "light", "ac", "heater", "dehum", "hum"]].values.astype(np.float32)
    first_row[0] /= 40.0
    first_row[1] /= 100.0
    first_row[2] /= 1000.0

    s_latent =  encoder(first_row[np.newaxis, np.newaxis, :]).numpy()[0]  # (FEATURE_DIM,)

    print("\nTest rollout:")
    for t in range(steps):
        qv = q_net.predict(s_latent.reshape(1, -1), verbose=0)[0]
        a = int(np.argmax(qv))
        bits = q_net.action_int_to_bits(a)

        next_idx = min(t + 1, len(df) - 1)
        next_row = df.iloc[next_idx][["temp", "humid", "light", "ac", "heater", "dehum", "hum"]].values.astype(np.float32)
        next_row[0] /= 40.0
        next_row[1] /= 100.0
        next_row[2] /= 1000.0
        next_row[3:3 + NUM_SWITCH] = bits

        # FIX: 用 next_row 计算 next_latent（而不是误用 s_raw）
        next_lat =  encoder(next_row[np.newaxis, np.newaxis, :]).numpy()[0]

        label_next = int(df.iloc[next_idx]["label"])
        r = q_net.compute_reward_batch(np.array([label_next]), np.array([bits]))[0]
        print(f"t={t:02d} action={a:02d} bits={bits.tolist()} reward={r:.3f} label_next={label_next}")
        s_latent = next_lat


class QModel_aug(QModel):
    def __init__(self):
        super().__init__()
        #self.meta_model = keras.models.load_model("meta_model.h5")
        #self.hvac_dense_layer = self.meta_model.get_layer("hvac_dense")  # lstm_encoder classifier
        #self.encoder = keras.models.Model(inputs=self.meta_model.input, outputs=self.hvac_dense_layer.output)
        self.encoder  = keras.models.load_model("encoder.h5")
        for layer in self.encoder .layers:
            layer.trainable = False

    #@tf.function(input_signature=[tf.TensorSpec([None, SEQ_LEN, NUM_FEATURES], tf.float32)])
    #def encode (self,x):
    #    return self.encoder (x, training=False)

    #@tf.function(input_signature=[tf.TensorSpec([None, SEQ_LEN, NUM_FEATURES], tf.float32)])
    def encode(self, x):
        # 自动调整输入形状
        x = np.array(x, dtype=np.float32)
        if x.ndim > 3:
            x = np.squeeze(x)
        if x.ndim == 2:  # (10, 7)
            x = np.expand_dims(x, axis=0)  # (1, 10, 7)
        latent=self.encoder(x)
        return tf.cast(latent, tf.float32)


    def encode_sequencex(self,encoder, X_seq):
        """
        Encode an entire sequence in batch.
        X_seq shape: (T, SEQ_LEN, NUM_FEATS) 或 (batch, SEQ_LEN, NUM_FEATS)
        Returns: (T, FEATURE_DIM)
        """
        if len(X_seq.shape) == 2:
            # 单样本 (SEQ_LEN, NUM_FEATS) -> (1, SEQ_LEN, NUM_FEATS)
            X_seq = np.expand_dims(X_seq, axis=0)

        # encoder 输出 (batch, FEATURE_DIM)
        z = self.encode(X_seq).numpy()
        return z  # shape: (batch, FEATURE_DIM)

    def encode_per_timestep(self, X_seq=None, seq_len=10, step_size=30):
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
        latents = self.encode(all_windows.astype(np.float32)).numpy()  # (num_windows, FEATURE_DIM)

        # 创建一个零矩阵，用来存储最终的 latents 序列
        latents_full = np.zeros((T, latents.shape[-1]), dtype=np.float32)

        # 填充 latent 序列
        for t in range(len(latents)):
            latents_full[t * step_size: t * step_size + seq_len] = latents[t]

        return latents_full


    # ------------------ 完全批量化 SARSA 训练 ------------------
    def sarsa_full_batch_robust(self,seq_len,files,num_feats,eps=0.1):

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
                s_latent_seq = self.encode_per_timestep( raw_seq,seq_len,num_feats)
                # Debug the input shape
                #print(f"Input shape: {s_latent_seq[:-1].shape}")
                #print(f"Expected input shape: {q_net.input_shape}")

                # Minimal fix - just ensure q_pred is always defined
                # 降維到 (10, 16)
                #s_latent_tra = tf.transpose(s_latent_seq)
                proj_layer = keras.layers.Dense(16, activation=None)
                s_latent_pred = proj_layer(s_latent_seq)  # shape -> (batch, seq_len, 16)
                s_latent_pred = tf.cast(s_latent_pred, tf.float32)

                q_pred = self.q_net.predict(s_latent_pred[:-1], verbose=0)

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

                s_next_latent_seq = self.encode_per_timestep( raw_next_seq,seq_len,num_feats)

                # 降維到 (10, 16)

                #s_next_latent_tra = tf.transpose(s_next_latent_seq)
                proj_layer_next = keras.layers.Dense(16, activation=None)
                s_next_latent_pred = proj_layer_next(s_next_latent_seq)
                s_next_latent_pred = tf.cast(s_next_latent_pred, tf.float32)

                # Q 預測
                q_pred_next = self.q_net.predict(s_next_latent_pred, verbose=0)  # shape: (seq_len-1, num_actions)
                a_next = self.select_action_batch(q_pred_next, eps)  # shape: (seq_len-1,)

                rewards = self.compute_reward_batch(labels[1:], actions_bits)
                total_reward += np.sum(rewards)


                self.sarsa_batch_update(
                     s_latent=s_latent_pred[:-1] ,a=actions,r=rewards,s_next_latent=s_next_latent_pred,a_next=a_next)
                #self.sarsa_batch_update(
                #    tf.convert_to_tensor(s_latent_seq[:-1], dtype=tf.float32),
                #    tf.convert_to_tensor(actions, tf.int32),
                #    tf.convert_to_tensor(s_next_latent_seq, dtype=tf.float32),
                #    tf.convert_to_tensor(rewards, tf.float32),
                #    a_next
                #)

            eps = max(EPS_END, eps * EPS_DECAY)
            if ep % 10 == 0 or ep == 1:
                print(f"sarsa_full_batch_robust {ep}/{SARSA_EPISODES}  eps={eps:.3f}  total_reward={total_reward:.3f}")



# ------------------ 混合精度 ------------------
from tensorflow.keras import mixed_precision
def main(args):

    policy = mixed_precision.Policy('mixed_float16') #mixed_float16
    mixed_precision.set_global_policy(policy)

    eps = EPS_START


    files = [os.path.join(args.load_dir, f) for f in os.listdir(args.load_dir) if f.endswith(".csv")]

    df_all = load_all_csvs(args.load_dir)
    X = df_all[["temp", "humid", "light", "ac", "heater", "dehum", "hum"]].values.astype(np.float32)
    y = df_all["label"].values.astype(np.int32)
    X[:, 0] /= 40.0
    X[:, 1] /= 100.0
    X[:, 2] /= 1000.0
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)
    qmodel = QModel_aug()  # build_q_network()

    #qmodel.model.summary()


    #qmodel=QModel()
    qmodel.sarsa_full_batch_robust( files=files, seq_len=SEQ_LEN, num_feats=NUM_FEATURES)
    save_sarsa_tflite(qmodel.q_net)
    #test_rollout(encoder)


    #qmodel.rollout_meta_sarsa(X_train, X_val, qmodel.encoder, SEQ_LEN, NUM_FEATURES, steps=30, epsilon=0.1)

# ------------------ 主程序 ------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder_mode", type=str, default=ENCODER_MODE, choices=["finetune", "freeze", "last_n"])
    parser.add_argument("--load_dir", type=str, default=DATA_DIR)
    parser.add_argument("--meta_out_tf", type=str, default=META_OUT_TF)
    parser.add_argument("--last_n", type=int, default=LAST_N)
    args, _ = parser.parse_known_args()

    ENCODER_MODE = args.encoder_mode
    LOAD_DIR = args.load_dir
    LAST_N = args.last_n
    META_OUT_TF = args.meta_out_tf
    #train_pipeline(DATA_DIR, tflite_out=META_OUT_TF)
    main(args)
