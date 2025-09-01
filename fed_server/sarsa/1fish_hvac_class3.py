import os
import numpy as np
import pandas as pd
import argparse
from utils_module import *

from global_hyparm import *

def main():
    # ==== 批量生成 ====
    #save_glob =os.path.join(DATA_DIR, f"*.csv") # "../../../../data/sarsa/*.csv"

    for i in range(NUM_FILES):
        df = generate_plant_sequence(DATA_DIR,BSEQ_LEN, NOISE_STD)
        file_path = os.path.join(DATA_DIR, f"plant_seq_with_hvac_fail_{i}.csv")
        df.to_csv(file_path, index=False)
        print(f"Saved {file_path}, shape: {df.shape}")

    # ------------------ 数据生成 ------------------
    os.makedirs(DATA_DIR, exist_ok=True)
    for i in range(NUM_FILES):
        file_path = os.path.join(DATA_DIR, f"plant_seq_with_hvac_fail_{i}.csv")
        if not os.path.exists(file_path):
            df = generate_plant_sequence(DATA_DIR, seq_len=SEQ_LEN, noise_std=NOISE_STD)
            df.to_csv(file_path, index=False)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", type=str, default=DATA_DIR)
    parser.add_argument("--num_files", type=int, default=NUM_FILES)
    parser.add_argument("--seq_len", type=int, default=SEQ_LEN)
    args, _ = parser.parse_known_args()

    DATA_DIR = args.save_dir
    NUM_FILES = args.num_files
    SEQ_LEN = args.seq_len
    main()