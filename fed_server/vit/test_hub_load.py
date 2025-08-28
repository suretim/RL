import tensorflow as tf
import tensorflow_hub as hub

# 自动处理ViT模型下载（首次运行需联网）
vit = hub.load("https://tfhub.dev/sayakpaul/vit_b16_fe/1")

# 测试随机输入（无需预先下载图像）
dummy_img = tf.random.uniform((1, 224, 224, 3))  # 模拟1张224x224的RGB图像
features = vit(dummy_img)  # 自动处理图像并输出特征

print(f"ViT特征向量形状: {features.shape}")  # 应为(1, 768)