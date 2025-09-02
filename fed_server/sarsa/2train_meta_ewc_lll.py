#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
End-to-end LSTM Meta-learning + EWC + Replay pipeline.
- Loads CSVs
- Contrastive pretrain
- Meta-learning (FOMAML)
- EWC regularization
- Replay buffer
- TFLite export
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))  # RL 根目录
DATA_DIR  = "../../../data/sarsa"
META_OUT_TF="../meta_model.tflite"
from tensorflow.keras import layers, models, optimizers
import argparse
from tensorflow.keras.optimizers import legacy
from global_hyparm import *
from utils_module import *
from utils_fisher import *

# ---------------- Hyperparameters ----------------
# =============================
# Hyperparameters
# =============================



ENCODER_MODE = "freeze"  # one of {"finetune","freeze","last_n"}
LAST_N = 1



# ------ Service (orchestrates the pipeline) ------
def serv_pipline(tflite_out,num_classes=3,seq_len=100, num_feats=7,feature_dim=64):
    #global ENCODER_MODE, LAST_N, DATA_GLOB, NUM_FEATS
    #ENCODER_MODE = encoder_mode
    #LAST_N = last_n
    #DATA_GLOB = data_glob

    model = MetaModel(num_classes=num_classes,seq_len=seq_len, num_feats=num_feats,feature_dim=feature_dim)
    # ===== Load data =====
    load_glob = os.path.join(LOAD_DIR, f"*.csv")
    X_unlabeled, X_labeled, y_labeled, num_feats = load_csv_data(load_glob,seq_len)

    # ===== Build encoder =====
    lstm_encoder = model.build_lstm_encoder()

    # ===== Contrastive pretraining =====
    contrastive_opt = tf.keras.optimizers.Adam()
    #contrastive_opt = legacy.Adam(META_LR)
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
            tasks = sample_tasks(X_labeled, y_labeled,num_tasks=5)
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
    if X_labeled.size > 0:
        model.save_tflite(meta_model, tflite_out)
        model.encoder.save("encoder.h5")
        model.classifier.save("classifier.h5")
        model.model.save("meta_model.h5")
        model.model.summary()
        print("save meta tflite Done.")

    # quick forward test
    dummy_x = np.random.rand(1, seq_len,num_feats).astype(np.float32)
    dummy_y = lstm_encoder(dummy_x)
    print("✅ Forward test shape:", dummy_y.shape)



# ---------------- End-to-End Serv Pipeline "meta_model_lstm.tflite" ----------------
def serv_train_tf(tflite_out,num_classes=3,seq_len=100, num_feats=7,feature_dim=64):
    print("Loading CSV data...")
    load_glob = os.path.join(LOAD_DIR, f"*.csv")
    X_unlabeled, X_labeled, y_labeled = load_csvs(load_glob,seq_len,num_feats)
    print("Unlabeled samples:", len(X_unlabeled))
    print("Labeled samples:", len(X_labeled), "Labels:", set(y_labeled))
    memory = ReplayBuffer()
    meta_model = MetaModel()
    # optimizer = optimizers.Adam(META_LR)
    optimizer = legacy.Adam(META_LR)

    # ---------------- Contrastive Pretrain ----------------
    print("Start contrastive pretrain...")
    anchors, positives = MetaModel.make_contrastive_pairs(X_unlabeled)
    dataset = tf.data.Dataset.from_tensor_slices((anchors, positives)).batch(BATCH_SIZE)
    c_loss = NTXentLoss()
    for ep in range(EPOCHS_CONTRASTIVE):
        epoch_loss = []
        for a, p in dataset:
            with tf.GradientTape() as tape:
                z_a = meta_model.encoder(a, training=True)
                z_p = meta_model.encoder(p, training=True)
                loss = c_loss(z_a, z_p)
            grads = tape.gradient(loss, meta_model.encoder.trainable_variables)
            optimizer.apply_gradients(zip(grads, meta_model.encoder.trainable_variables))
            epoch_loss.append(loss.numpy())
        print(f"[Contrastive] Epoch {ep + 1}/{EPOCHS_CONTRASTIVE}, loss={np.mean(epoch_loss):.4f}")

    # ---------------- Meta-learning ----------------
    print("Start meta-learning...")


    fisher_matrix = MetaModel.compute_fisher_matrix(meta_model.model, X_labeled, y_labeled)
    if fisher_matrix is not None:
        prev_weights = deepcopy(meta_model.model.trainable_variables)
        for ep in range(EPOCHS_META):
            tasks = sample_tasks(X_labeled, y_labeled)
            #prev_weights = deepcopy(meta_model.model.trainable_variables)
            #fisher_matrix = MetaModel.compute_fisher_matrix(meta_model, X_labeled, y_labeled)

            loss, acc, _ = meta_model.outer_update_with_lll(
                memory, meta_model.model, optimizer, tasks,
                prev_weights=prev_weights,
                fisher_matrix=fisher_matrix,
                lambda_ewc=0.4  # 🔑 这里可以调节强度
            )
            print(f"[Meta] Epoch {ep + 1}/{EPOCHS_META}, query_loss={loss:.4f}, query_acc={acc:.4f}")

        MetaModel.save_fisher_and_weights(model=meta_model.model, fisher_matrix=fisher_matrix)


    # ---------------- Export TFLite ----------------
    MetaModel.save_tflite(meta_model.model, tflite_out)
    meta_model.save("meta_model_lstm.h5")
    # ===== Save assets =====


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
    META_OUT_TF=args.meta_out_tf
    #serv_train_tf(META_OUT_TF,num_classes=NUM_CLASSES,seq_len=SEQ_LEN, num_feats=NUM_FEATURES,feature_dim=FEATURE_DIM)
    serv_pipline(META_OUT_TF, num_classes=NUM_CLASSES,seq_len=SEQ_LEN, num_feats=NUM_FEATURES,feature_dim=FEATURE_DIM)
