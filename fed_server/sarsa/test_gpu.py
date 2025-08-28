import tensorflow as tf

print("TensorFlow version:", tf.__version__)

# 检查 GPU 是否可用
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    print("✅ GPU(s) detected:", gpus)
    # 打印详细 GPU 信息
    for gpu in gpus:
        details = tf.config.experimental.get_device_details(gpu)
        print("   -", details.get("device_name", "Unknown GPU"))
else:
    print("❌ No GPU detected, running on CPU")

# 检查混合精度策略
from tensorflow.keras import mixed_precision
policy = mixed_precision.global_policy()
print("Current mixed precision policy:", policy)

# 如果 GPU 支持，测试 float16
if gpus:
    try:
        with tf.device("/GPU:0"):
            a = tf.constant([1.0, 2.0, 3.0], dtype=tf.float16)
            b = tf.constant([4.0, 5.0, 6.0], dtype=tf.float16)
            print("float16 test result:", tf.add(a, b))
    except Exception as e:
        print("float16 test failed:", e)
