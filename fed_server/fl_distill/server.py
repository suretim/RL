#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Flower Server for Federated KD with Global Soft Labels
- The server maintains a TEACHER model (e.g., a big CNN/ViT exported as .h5).
- Each round:
  1) Server loads a small PUBLIC dataset (shared path on all nodes).
  2) Runs teacher to produce soft labels with temperature T.
  3) Broadcasts the soft labels (and class_count) to clients via 'fit_ins.config'.
- Clients train students with CE(on private data) + KL(on public data & soft labels).

Run:
  python server.py --teacher_path ./teacher.h5 --public_dir ./public_ds --img_size 224 224 \
                   --rounds 5 --temperature 2.0

Install:
  pip install flwr tensorflow
"""

import argparse
import os
import json
import base64
import numpy as np
import tensorflow as tf
import flwr as fl

AUTOTUNE = tf.data.AUTOTUNE

def load_public_dataset(public_dir, img_size, batch_size):
    ds = tf.keras.utils.image_dataset_from_directory(
        public_dir, labels="inferred", label_mode="int",
        image_size=img_size, batch_size=batch_size, shuffle=False)
    class_names = ds.class_names
    num_classes = len(class_names)
    def norm(x, y): return (tf.image.convert_image_dtype(x, tf.float32), y)
    ds = ds.map(norm, num_parallel_calls=AUTOTUNE)
    return ds, num_classes, class_names

def run_teacher_softlabels(teacher, dataset, temperature):
    """Return tuple (indices, soft_probs) aligned with dataset order."""
    all_soft = []
    all_idx = []
    idx = 0
    for x, _ in dataset:
        logits = teacher(x, training=False)
        soft = tf.nn.softmax(logits / temperature, axis=-1).numpy()
        b = soft.shape[0]
        all_soft.append(soft)
        all_idx.extend(list(range(idx, idx + b)))
        idx += b
    soft_all = np.concatenate(all_soft, axis=0)
    return np.array(all_idx, dtype=np.int32), soft_all

def ndarray_to_b64(arr: np.ndarray) -> str:
    # Compact transfer: float32 -> bytes -> base64
    return base64.b64encode(arr.tobytes()).decode("utf-8")

def b64_meta(shape, dtype):
    return {"shape": list(shape), "dtype": str(np.dtype(dtype))}

class SoftLabelStrategy(fl.server.strategy.FedAvg):
    """FedAvg for student weights, plus broadcasting teacher soft labels each round."""

    def __init__(self, teacher_path, public_dir, img_size, batch_size, temperature, **kwargs):
        super().__init__(**kwargs)
        self.teacher = tf.keras.models.load_model(teacher_path, compile=False)
        self.public_dir = public_dir
        self.img_size = tuple(img_size)
        self.batch_size = batch_size
        self.temperature = float(temperature)

        # Preload public dataset once (order must be consistent across server/clients)
        self.public_ds, self.num_classes, self.class_names = load_public_dataset(
            public_dir=self.public_dir, img_size=self.img_size, batch_size=self.batch_size
        )

    def configure_fit(self, server_round, parameters, client_manager):
        # Produce global soft labels on public dataset
        idx_arr, soft_arr = run_teacher_softlabels(self.teacher, self.public_ds, self.temperature)

        # Serialize arrays
        payload = {
            "public_len": int(soft_arr.shape[0]),
            "num_classes": int(soft_arr.shape[1]),
            "temperature": self.temperature,
            "indices": ndarray_to_b64(idx_arr.astype(np.int32)),
            "indices_meta": b64_meta(idx_arr.shape, np.int32),
            "soft_b64": ndarray_to_b64(soft_arr.astype(np.float32)),
            "soft_meta": b64_meta(soft_arr.shape, np.float32),
        }
        cfg = {"kd_payload": json.dumps(payload)}

        # Default FedAvg behavior for selecting clients
        fit_ins_list = super().configure_fit(server_round, parameters, client_manager)

        # Inject config to each client's FitIns
        new_fit_ins_list = []
        for client, fit_ins in fit_ins_list:
            new_fit_ins = fl.common.FitIns(parameters=fit_ins.parameters, config={**fit_ins.config, **cfg})
            new_fit_ins_list.append((client, new_fit_ins))
        return new_fit_ins_list

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--teacher_path", type=str, required=True)
    p.add_argument("--public_dir", type=str, required=True, help="shared small public dataset")
    p.add_argument("--img_size", nargs=2, type=int, default=[224, 224])
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--rounds", type=int, default=5)
    p.add_argument("--temperature", type=float, default=2.0)
    args = p.parse_args()

    strategy = SoftLabelStrategy(
        teacher_path=args.teacher_path,
        public_dir=args.public_dir,
        img_size=args.img_size,
        batch_size=args.batch_size,
        temperature=args.temperature,
        min_fit_clients=2,
        min_available_clients=2,
        fraction_fit=1.0,
        fraction_evaluate=0.0,
    )

    fl.server.start_server(
        server_address="0.0.0.0:8080",
        strategy=strategy,
        config=fl.server.ServerConfig(num_rounds=args.rounds),
    )

if __name__ == "__main__":
    main()
