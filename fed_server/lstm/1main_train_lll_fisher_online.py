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
- V1.0
"""

#import os, glob
#import numpy as np
#import pandas as pd
#import tensorflow as tf
#from tensorflow.keras import layers, models
#import random
import argparse 
import datetime
import json
from util_model import *
#from utils_fisher import *
# =============================
# Hyperparameters
# =============================
DATA_GLOB = "../../../data/lll_data/*.csv"
SEQ_LEN = 10
FEATURE_DIM = 64
BATCH_SIZE = 32
EPOCHS_CONTRASTIVE = 10
EPOCHS_META = 20
INNER_LR = 1e-2
META_LR = 1e-3
NUM_CLASSES = 3
NUM_TASKS = 5
SUPPORT_SIZE = 10
QUERY_SIZE = 20
REPLAY_CAPACITY = 1000
REPLAY_WEIGHT = 0.3
LAMBDA_EWC = 1e-3
NUM_FEATS=7
#ENCODER_MODE = finetune  freeze last_n
ENCODER_MODE ="freeze"
LAST_N= 1
FLOWERING_WEIGHT = 2.0  # gradient boost upper bound for flowering-focus


if __name__ == "__main__":
    parser = argparse.ArgumentParser() 
    parser.add_argument("--encoder_mode", type=str, default=ENCODER_MODE)
    parser.add_argument("--data_glob", type=str, default=DATA_GLOB)
    parser.add_argument("--last_n", type=int, default=LAST_N)
    #args = parser.parse_args()
    # ============ 解析參數 ============
    args, unknown = parser.parse_known_args()

    meta_model = MetaModel(feature_dim=FEATURE_DIM)
    meta_model.service_pipeline(encoder_mode=args.encoder_mode,last_n=args.last_n,data_glob=args.data_glob)

