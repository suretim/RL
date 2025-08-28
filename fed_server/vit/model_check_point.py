import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# 示例数据（实际替换为你的数据）
(x_train, y_train), (x_val, y_val) = tf.keras.datasets.mnist.load_data()
x_train, x_val = x_train / 255.0, x_val / 255.0  # 归一化

# 定义一个简单模型
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 定义 ModelCheckpoint 回调（核心部分）
checkpoint_path = 'best_model.tf'  # 推荐使用 .keras 格式避免 HDF5 问题
callbacks = [
    ModelCheckpoint(
        filepath=checkpoint_path,
        monitor='val_accuracy',      # 监控验证集准确率
        mode='max',                  # 取最大值（因为准确率越高越好）
        save_best_only=True,         # 只保存最佳模型
        verbose=1                    # 打印保存日志
    ),
    EarlyStopping(monitor='val_accuracy', patience=3)  # 提前停止
]

# 训练模型
history = model.fit(
    x_train, y_train,
    epochs=50,
    validation_data=(x_val, y_val),
    callbacks=callbacks
)

# 训练完成后，加载最佳模型进行推理
best_model = tf.keras.models.load_model(checkpoint_path)
val_loss, val_acc = best_model.evaluate(x_val, y_val)
print(f"\nBest model validation accuracy: {val_acc:.4f}")