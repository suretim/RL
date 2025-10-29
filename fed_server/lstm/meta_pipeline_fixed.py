#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Meta-learning pipeline with HVAC-aware features and flowering-period focus.
- Expects CSV columns: temp, humid, light, ac, heater, dehum, hum, label
- Sliding windows -> contrastive learning (unlabeled) + FOMAML with LLL + EWC (labeled)
- Encoder: LSTM on continuous features only (temp/humid/light)
- Additional HVAC features: mean on/off rate + toggle rate (abs(diff)) over time
- Gradient boost on flowering period with abnormal HVAC toggling
- TFLite export restricted to TFLITE_BUILTINS
- Includes Fisher matrix computation and loading for EWC
- V1.1 (cleaned & runnable)
"""


from util_model import *


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder_mode", type=str, default=ENCODER_MODE, choices=["finetune", "freeze", "last_n"])
    parser.add_argument("--data_glob", type=str, default=DATA_GLOB)
    parser.add_argument("--last_n", type=int, default=LAST_N)
    args, _ = parser.parse_known_args()

    ENCODER_MODE = args.encoder_mode
    DATA_GLOB = args.data_glob
    LAST_N = args.last_n

    meta = MetaModel(feature_dim=FEATURE_DIM)
    meta.service_pipeline(encoder_mode=ENCODER_MODE, last_n=LAST_N, data_glob=DATA_GLOB)
