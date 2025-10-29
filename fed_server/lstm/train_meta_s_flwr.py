# 3sarsa_controller.py
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from sklearn.model_selection import train_test_split

# ---------------------
# 參數
# ---------------------
SAVE_DIR = "../../../data/lll_data"   # 會在本腳本下創建資料夾
NUM_FILES = 30
SEQ_LEN = 1000
NOISE_STD = 0.5

ENCODER_LATENT_DIM = 16
Q_HIDDEN = [64, 64]
ACTION_COST = 0.05

SARSA_EPISODES = 200      # episode 數（每 episode 隨機挑一個 csv）
GAMMA = 0.95
EPS_START = 0.3
EPS_END = 0.05
EPS_DECAY = 0.995
LR_Q = 1e-3
BATCH_PRINT_EVERY = 10

# ---------------------
# 你的數據生成器（直接採用、少量修改）
# ---------------------
def generate_plant_sequence(seq_len=1000, noise_std=0.5, insect_prob=0.3, equip_fail_prob=0.2):
    t, h, l, labels = [], [], [], []
    acs, heaters, dehums, hums = [], [], [], []

    insect_event = np.random.rand() < insect_prob
    insect_start = np.random.randint(300, 800) if insect_event else -1
    insect_end   = insect_start + np.random.randint(50, 150) if insect_event else -1

    equip_fail_event = np.random.rand() < equip_fail_prob
    fail_type = None
    fail_start, fail_end = -1, -1
    if equip_fail_event:
        fail_type = np.random.choice(["humidifier_fail", "dehumidifier_fail", "heater_fail", "ac_fail"])
        fail_start = np.random.randint(200, 700)
        fail_end = fail_start + np.random.randint(80, 200)

    for step in range(seq_len):
        if step < 200:   # 育苗期
            base_t, base_h, base_l = 22, 65, 250
        elif step < 600: # 生长期
            base_t, base_h, base_l = 25, 58, 400
        else:            # 开花期
            base_t, base_h, base_l = 28, 48, 600

        ti = base_t + np.sin(step/50) + np.random.randn() * noise_std
        hi = base_h + np.cos(step/70) + np.random.randn() * noise_std
        li = base_l + np.sin(step/100) * 20 + np.random.randn() * noise_std * 5

        # default label
        if (ti < 10) or (li < 100):
            label = 1
        elif (ti < 15) or (ti > 35) or (hi < 30) or (hi > 80) or (li > 800):
            label = 2
        else:
            label = 0

        if insect_event and insect_start <= step <= insect_end:
            li *= np.random.uniform(0.6, 0.8)
            hi += np.random.uniform(-5, 5)
            label = 2

        # HVAC default logic (the generator's internal control)
        ac_state = 1 if ti > 26 else 0
        heater_state = 1 if ti < 20 else 0
        dehum_state = 1 if hi > 70 else 0
        hum_state = 1 if hi < 40 else 0

        # equipment fail period
        if equip_fail_event and fail_start <= step <= fail_end:
            if fail_type == "humidifier_fail":
                hum_state = 1
                hi += np.random.uniform(5, 15)
                label = 2
            elif fail_type == "dehumidifier_fail":
                dehum_state = 1
                hi -= np.random.uniform(5, 15)
                label = 2
            elif fail_type == "heater_fail":
                heater_state = 1
                ti += np.random.uniform(5, 10)
                label = 2
            elif fail_type == "ac_fail":
                ac_state = 1
                ti -= np.random.uniform(5, 10)
                label = 2

        # small disturbance under unhealthy period
        if label == 2:
            if np.random.rand() < 0.05: ac_state = 1 - ac_state
            if np.random.rand() < 0.05: heater_state = 1 - heater_state
            if np.random.rand() < 0.05: dehum_state = 1 - dehum_state
            if np.random.rand() < 0.05: hum_state = 1 - hum_state

        t.append(ti); h.append(hi); l.append(li)
        labels.append(label)
        acs.append(ac_state); heaters.append(heater_state); dehums.append(dehum_state); hums.append(hum_state)

    return pd.DataFrame({
        "temp": t,
        "humid": h,
        "light": l,
        "ac": acs,
        "heater": heaters,
        "dehum": dehums,
        "hum": hums,
        "label": labels
    })

# ---------------------
# 產生資料（若已經有 csv 可略過）
# ---------------------
os.makedirs(SAVE_DIR, exist_ok=True)
print("Generating CSV files...")
for i in range(NUM_FILES):
    file_path = os.path.join(SAVE_DIR, f"plant_seq_with_hvac_fail_{i}.csv")
    if not os.path.exists(file_path):
        df = generate_plant_sequence(SEQ_LEN, NOISE_STD)
        df.to_csv(file_path, index=False)
        print(f"Saved {file_path}, shape: {df.shape}")
print("Data generation done.")

# ---------------------
# 載入所有 CSV 做 supervised pretrain（classifier -> encoder）
# ---------------------
def load_all_csvs(data_dir):
    files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".csv")]
    dfs = []
    for f in files:
        dfs.append(pd.read_csv(f))
    return pd.concat(dfs, axis=0).reset_index(drop=True)

print("Loading CSVs for pretrain...")
df_all = load_all_csvs(SAVE_DIR)
X = df_all[["temp","humid","light","ac","heater","dehum","hum"]].values.astype(np.float32)
y = df_all["label"].values.astype(np.int32)

# Normalize continuous features
X[:,0] = X[:,0] / 40.0        # temp normalize
X[:,1] = X[:,1] / 100.0       # humid
X[:,2] = X[:,2] / 1000.0      # light

# train-test split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)

# classifier model (we will use the penultimate layer as encoder)
def build_classifier(input_dim=7, latent_dim=ENCODER_LATENT_DIM):
    inp = layers.Input(shape=(input_dim,))
    x = layers.Dense(64, activation="relu")(inp)
    x = layers.Dense(64, activation="relu")(x)
    latent = layers.Dense(latent_dim, activation="relu", name="latent")(x)
    out = layers.Dense(3, activation="softmax")(latent)
    model = models.Model(inp, out)
    return model

print("Building classifier and pretraining encoder...")
clf = build_classifier()
clf.compile(optimizer=optimizers.Adam(1e-3), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
clf.fit(X_train, y_train, validation_data=(X_val,y_val), epochs=8, batch_size=512, verbose=2)

# Extract encoder (take input -> latent)
encoder = models.Model(clf.input, clf.get_layer("latent").output)
print("Encoder summary:")
encoder.summary()

# ---------------------
# SARSA with function approximation (Q-network)
# ---------------------
# action mapping helpers: 4 binary HVAC bits -> integer 0..15
def action_int_to_bits(a_int):
    bits = [(a_int >> i) & 1 for i in range(4)]  # ac, heater, dehum, hum (LSB->MSB but consistent)
    return np.array(bits, dtype=np.float32)

def bits_to_action_int(bits):
    val = 0
    for i, b in enumerate(bits):
        val |= (int(b) << i)
    return val

NUM_ACTIONS = 16
LATENT_DIM = ENCODER_LATENT_DIM

def build_q_network(latent_dim=LATENT_DIM, num_actions=NUM_ACTIONS):
    inp = layers.Input(shape=(latent_dim,))
    x = inp
    for h in Q_HIDDEN:
        x = layers.Dense(h, activation="relu")(x)
    out = layers.Dense(num_actions, activation="linear")(x)
    model = models.Model(inp, out)
    return model

q_net = build_q_network()
q_optimizer = optimizers.Adam(LR_Q)
mse_loss = tf.keras.losses.MeanSquaredError()

# utility: epsilon-greedy
def select_action(q_values, eps):
    if np.random.rand() < eps:
        return np.random.randint(NUM_ACTIONS)
    return int(np.argmax(q_values))

# reward function
def compute_reward(label, action_bits):
    if label == 0:
        r = 1.0
    elif label == 1:
        r = -1.0
    else:
        r = -2.0
    r -= ACTION_COST * np.sum(action_bits)  # penalty for turning devices on
    return float(r)

# training loop: SARSA online updates
eps = EPS_START
files = [os.path.join(SAVE_DIR, f) for f in os.listdir(SAVE_DIR) if f.endswith(".csv")]
print(f"Found {len(files)} csv files for SARSA episodes.")

for ep in range(1, SARSA_EPISODES+1):
    # pick a random sequence file as environment
    df = pd.read_csv(np.random.choice(files))
    T = len(df)
    # initial step 0
    row0 = df.iloc[0].copy()
    # we let agent control HVAC; set initial HVAC to generator's value for step 0
    s0_raw = row0[["temp","humid","light","ac","heater","dehum","hum"]].values.astype(np.float32)
    # normalize continuous features same as pretrain
    s0_raw[0] /= 40.0; s0_raw[1] /= 100.0; s0_raw[2] /= 1000.0
    s0_latent = encoder.predict(s0_raw.reshape(1,-1), verbose=0)[0]

    # compute Q(s0)
    q_vals = q_net.predict(s0_latent.reshape(1,-1), verbose=0)[0]
    a0 = select_action(q_vals, eps)
    a0_bits = action_int_to_bits(a0)

    total_reward = 0.0

    for t in range(T-1):
        # apply action a_t onto next step's HVAC features to simulate control effect
        # read current step's label for reward next
        label_next = int(df.iloc[t+1]["label"])

        # take action a0 now; we consider environment transitions to t+1 and then agent will choose a1
        # build next state's raw features from df row t+1 but overwrite HVAC columns with action bits
        next_row = df.iloc[t+1].copy()
        next_row_vals = next_row[["heater","temp","humid","light","ac","dehum","hum"]].values.astype(np.float32)
        next_row_vals[0] /= 40.0; next_row_vals[1] /= 100.0; next_row_vals[2] /= 1000.0
        # overwrite HVAC (indices 3..6) with agent action bits
        next_row_vals[3:7] = a0_bits

        s_next_latent = encoder.predict(next_row_vals.reshape(1,-1), verbose=0)[0]

        # select a1 (policy on s_next)
        q_next_vals = q_net.predict(s_next_latent.reshape(1,-1), verbose=0)[0]
        a1 = select_action(q_next_vals, eps)
        a1_bits = action_int_to_bits(a1)

        # compute reward using next label (you could use immediate reward on current label; choose convention)
        r = compute_reward(label_next, a0_bits)
        total_reward += r

        # SARSA update: target = r + gamma * Q(s_next, a1)
        with tf.GradientTape() as tape:
            q_pred_all = q_net(s0_latent.reshape(1,-1), training=True)  # shape (1, num_actions)
            q_pred = tf.squeeze(q_pred_all[0, a0])  # scalar
            q_next_all = q_net(s_next_latent.reshape(1,-1), training=False)
            q_next_a1 = tf.squeeze(q_next_all[0, a1])
            target = r + GAMMA * q_next_a1
            loss = mse_loss(tf.expand_dims(target,0), tf.expand_dims(q_pred,0))

        grads = tape.gradient(loss, q_net.trainable_variables)
        q_optimizer.apply_gradients(zip(grads, q_net.trainable_variables))

        # shift s, a <- s_next, a1
        s0_latent = s_next_latent
        a0 = a1
        a0_bits = a1_bits

    # episode end
    eps = max(EPS_END, eps * EPS_DECAY)
    if ep % BATCH_PRINT_EVERY == 0 or ep == 1:
        print(f"Episode {ep}/{SARSA_EPISODES}  eps={eps:.3f}  total_reward={total_reward:.3f}")

print("SARSA training finished.")

# ---------------------
# 小測試：讓 agent 在一個新序列上 run 並顯示前 30 steps 的動作與 reward
# ---------------------
test_df = pd.read_csv(np.random.choice(files))
s_raw = test_df.iloc[0][["temp","humid","light","ac","heater","dehum","hum"]].values.astype(np.float32)
s_raw[0]/=40.0; s_raw[1]/=100.0; s_raw[2]/=1000.0
s_latent = encoder.predict(s_raw.reshape(1,-1), verbose=0)[0]

print("\nTest rollout (first 30 steps):")
for t in range(30):
    qv = q_net.predict(s_latent.reshape(1,-1), verbose=0)[0]
    a = int(np.argmax(qv))
    bits = action_int_to_bits(a)
    next_row_vals = test_df.iloc[min(t+1, len(test_df)-1)][["temp","humid","light","ac","heater","dehum","hum"]].values.astype(np.float32)
    next_row_vals[0]/=40.0; next_row_vals[1]/=100.0; next_row_vals[2]/=1000.0
    next_row_vals[3:7] = bits
    next_lat = encoder.predict(next_row_vals.reshape(1,-1), verbose=0)[0]
    label_next = int(test_df.iloc[min(t+1, len(test_df)-1)]["label"])
    r = compute_reward(label_next, bits)
    print(f"t={t:02d} action={a:02d} bits={bits.tolist()} reward={r:.3f} label_next={label_next}")
    s_latent = next_lat
