# ===== NumPy & TensorFlow 版本兼容自动检测 =====
import os
import sys
import importlib
import subprocess

def ensure_numpy_tf_compat():
    try:
        import tensorflow as tf
        import numpy as np
        tf_version = tf.__version__
        np_version = np.__version__
        print("NumPy version: ", np_version)
        print("tf_version version: ", tf_version)

        # TensorFlow 2.16 及之前版本不支持 NumPy 2.x
        if np_version.startswith("2.") and tf_version < "2.17":
            print(f"[警告] 检测到 NumPy {np_version} 与 TensorFlow {tf_version} 不兼容，正在降级 NumPy...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "numpy==1.26.4", "--upgrade"])
            print("[完成] NumPy 已降级到 1.26.4，请重新运行脚本。")
            os.execv(sys.executable, [sys.executable] + sys.argv)  # 重启当前脚本
    except ImportError:
        print("[提示] 检测失败，可能未安装 TensorFlow 或 NumPy。")

ensure_numpy_tf_compat()
# ===== 版本检测结束 =====

#import tensorflow as tf
#import argparse
#import os
#import numpy as np
