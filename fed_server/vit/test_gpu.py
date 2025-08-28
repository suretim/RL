import tensorflow as tf

from tensorflow.python.platform import build_info
print("TensorFlow Version: ", tf.__version__)
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

#print("cuDNN Version:", build_info.cudnn_version())

# 列出所有 GPU 设备
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"GPU devices found: {len(gpus)}")
    for gpu in gpus:
        print(f"Device name: {gpu.name}")
else:
    print("No GPUs found.")

# 检查是否可以使用 GPU
if tf.config.list_physical_devices('GPU'):
    print("GPU is available.")
else:
    print("GPU is not available.")

# 配置 TensorFlow 使用 GPU
tf.config.experimental.set_memory_growth(tf.config.list_physical_devices('GPU')[0], True)

# 确认是否启用 GPU
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
