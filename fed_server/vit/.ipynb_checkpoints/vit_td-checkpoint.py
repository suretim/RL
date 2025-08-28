import tensorflow as tf
from tensorflow.keras import layers, models, losses, optimizers
import tensorflow_hub as hub
import numpy as np

# 加载 ViT-B16 作为教师模型，只提取特征，不训练
vit_url = "https://tfhub.dev/sayakpaul/vit_b16_fe/1"
vit_layer = hub.KerasLayer(vit_url, trainable=False)

IMG_SIZE = (224, 224)
FEATURE_DIM = 768  # ViT-B16 输出特征维度

# 学生模型：轻量 CNN 编码器，输出同维度特征
def create_student_encoder(input_shape=(224, 224, 3), feature_dim=FEATURE_DIM):
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(32, 3, strides=2, padding='same', activation='relu')(inputs)  # 112x112x32
    x = layers.Conv2D(64, 3, strides=2, padding='same', activation='relu')(x)       # 56x56x64
    x = layers.Conv2D(128, 3, strides=2, padding='same', activation='relu')(x)      # 28x28x128
    x = layers.GlobalAveragePooling2D()(x)                                          # (batch, 128)
    x = layers.Dense(feature_dim)(x)                                                # 映射到768维，匹配教师特征
    model = models.Model(inputs=inputs, outputs=x)
    return model

student_encoder = create_student_encoder(input_shape=IMG_SIZE+(3,), feature_dim=FEATURE_DIM)

# 蒸馏训练步骤
def distillation_train(student, teacher, train_ds, epochs=10):
    optimizer = optimizers.Adam()
    mse_loss_fn = losses.MeanSquaredError()

    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        for step, (images, _) in enumerate(train_ds):
            with tf.GradientTape() as tape:
                teacher_features = teacher(images, training=False)
                student_features = student(images, training=True)

                loss = mse_loss_fn(teacher_features, student_features)

            grads = tape.gradient(loss, student.trainable_weights)
            optimizer.apply_gradients(zip(grads, student.trainable_weights))

            if step % 10 == 0:
                print(f" Step {step}: distillation loss = {loss.numpy():.4f}")

# 准备训练数据（示例用随机数据替代）
def create_dummy_dataset(batch_size=32, num_batches=100):
    for _ in range(num_batches):
        imgs = np.random.rand(batch_size, *IMG_SIZE, 3).astype(np.float32)
        labels = np.zeros((batch_size,))  # 标签可忽略，蒸馏只用输入图像和教师特征
        yield imgs, labels

train_dataset = tf.data.Dataset.from_generator(
    create_dummy_dataset,
    output_types=(tf.float32, tf.float32),
    output_shapes=((32, *IMG_SIZE, 3), (32,))
)

# 执行蒸馏训练
distillation_train(student_encoder, vit_layer, train_dataset, epochs=5)

# 训练完成后，保存学生模型（轻量编码器）
student_encoder.save('student_encoder_distilled.h5')
