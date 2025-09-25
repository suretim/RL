import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import tensorflow_hub as hub
import numpy as np
from pathlib import Path

print("TensorFlow version:", tf.__version__)  # 确认版本为2.12+

# 数据配置
BATCH_SIZE = 32
IMG_SIZE = (224, 224)
EPOCHS = 1
categories = ["y", "w", "n"]  # 假设的类别

# 转换为绝对路径
current_dir = Path(__file__).parent.absolute()
#DATA_BASE_DIR= "E:/tim/vit_dataset"
DATA_BASE_DIR = "../../../../dataset/sprout_y_n_data3"
#DATA_BASE_DIR = current_dir / "vit_dataset"  # 改为您的实际目录名
# 从TensorFlow Hub加载ViT模型
vit_url = "https://tfhub.dev/sayakpaul/vit_b16_fe/1"
# 修改为您的实际数据集路径（重要！）
#DATA_BASE_DIR = "../../../../vit_dataset"  # 请确认这个路径是否正确
# 检查并创建数据集目录（调试用）
def verify_dataset_structure(base_dir):
    """验证数据集目录结构是否正确"""
    for split in ['train', 'val', 'test']:
        split_dir = os.path.join(base_dir, split)
        print(f"\nChecking {split_dir}:")
        if not os.path.exists(split_dir):
            print(f"❌ Directory missing: {split_dir}")
            continue

        for category in categories:
            cat_dir = os.path.join(split_dir, category)
            if os.path.exists(cat_dir):
                num_images = len([f for f in os.listdir(cat_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
                print(f"  {category}: {num_images} images")
            else:
                print(f"  ❌ Missing category directory: {cat_dir}")


verify_dataset_structure(DATA_BASE_DIR)


# 数据流生成
def create_dataflow(directory, augment=False):
    """创建数据流"""
    generator = train_datagen if augment else val_test_datagen
    return generator.flow_from_directory(
        directory,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=augment  # 仅训练集需要shuffle
    )

# ViT专用预处理（重要！）
def vit_preprocess(image):
    """ViT模型需要的特殊预处理"""
    image = tf.cast(image, tf.float32) / 255.0  # 归一化到[0,1]
    image = (image - 0.5) * 2.0  # 转换到[-1,1]范围
    return image

# 数据增强配置
train_datagen = ImageDataGenerator(
    preprocessing_function=vit_preprocess,  # 应用ViT预处理
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
)
val_test_datagen = ImageDataGenerator(
    preprocessing_function=vit_preprocess  # 验证/测试集使用相同预处理
)
val_data = create_dataflow(os.path.join(DATA_BASE_DIR, "val"))
train_data = create_dataflow(os.path.join(DATA_BASE_DIR, "train"), augment=True)
test_data = create_dataflow(os.path.join(DATA_BASE_DIR, "test"))

# 强制检查数据是否加载成功
assert train_data.samples > 0, f"训练集未找到图片！请检查路径: {os.path.join(DATA_BASE_DIR, 'train')}"
assert val_data.samples > 0, f"验证集未找到图片！请检查路径: {os.path.join(DATA_BASE_DIR, 'val')}"
print(f"\n数据加载成功！类别对应关系: {train_data.class_indices}")

def build_vit_model():
    """构建ViT分类模型"""
    vit_layer = hub.KerasLayer(
        vit_url,
        trainable=False,  # 微调时设为True
        name='vit_feature_extractor'
    )

    inputs = tf.keras.Input(shape=IMG_SIZE + (3,))
    x = vit_layer(inputs)  # ViT会自动处理预处理

    # 自定义分类头
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    outputs = tf.keras.layers.Dense(len(categories), activation='softmax')(x)

    return tf.keras.Model(inputs, outputs)


model = build_vit_model()
model.summary()



# 编译模型
#model.compile(
#    optimizer=tf.keras.optimizers.Adam(1e-4),
#    loss='categorical_crossentropy',
#    metrics=['accuracy']
#)
# 编译模型（定义优化器、损失和指标）
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# 用一批假数据跑一次，触发 trace

dummy_x = np.random.rand(1, IMG_SIZE[0], IMG_SIZE[1], 3).astype(np.float32)
dummy_y = np.zeros((1, len(categories)), dtype=np.float32)
dummy_y[0, 0] = 1  # 假标签
model.train_on_batch(dummy_x, dummy_y)

# 再保存
tf.saved_model.save(model, "saved_model_best", options=tf.saved_model.SaveOptions(experimental_custom_gradients=False))

#model.save("saved_model_best")
print(f"\n数据加载成功！类别对应关系: {train_data.class_indices}")




callbacks = [
    tf.keras.callbacks.EarlyStopping(
        patience=5,
        monitor='val_loss',  # 明确监控验证集损失（必须指定！）
        restore_best_weights=True
    ),
    tf.keras.callbacks.ModelCheckpoint(
        'best_model.tf',  # or .tf
        save_best_only=True,
        monitor='val_loss',  # 与EarlyStopping一致
        mode='min'          # 监控指标的方向（min表示损失越小越好）
        #overwrite=True      # 解决HDF5文件冲突的关键！
    )
]

# 训练模型
#print("\n开始训练...")
history = model.fit(
    train_data,
    epochs=EPOCHS,
    #validation_data=val_data
    callbacks=callbacks
)
tf.saved_model.save(model, "best_model", options=tf.saved_model.SaveOptions(experimental_custom_gradients=False))

#keras_model = tf.keras.models.load_model('best_model.tf')

#converter = tf.lite.TFLiteConverter.from_saved_model("saved_model")

#converter = tf.lite.TFLiteConverter.from_keras_model(model)
#tflite_model = converter.convert()
#with open('model.tflite', 'wb') as f:
#    f.write(tflite_model)



# 评估测试集
print("\n测试集评估:")
test_loss, test_acc = model.evaluate(test_data)
print(f"测试准确率: {test_acc:.4f}")