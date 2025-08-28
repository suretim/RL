import keras  # Keras 3
import tensorflow as tf
from keras.layers import TFSMLayer
from keras.models import Sequential

# 加载 SavedModel 作为推理层
model = Sequential([
    TFSMLayer("best_model.tf", call_endpoint="serving_default")
])

# 使用示例
import numpy as np
dummy_input = np.random.rand(1, 224, 224, 3)  # 替换为您的实际输入尺寸
output = model.predict(dummy_input)
# 加载 SavedModel 作为推理层
#model = TFSMLayer("best_model.tf", call_endpoint="serving_default")  # 注意检查 call_endpoint 名称
#model = keras.saving.load_model("best_model.tf")
#model = tf.keras.models.load_model("best_model.tf")
keras_model = tf.keras.models.load_model('best_model.tf')
converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
tflite_model = converter.convert()
with open('model.tflite', 'wb') as f:
    f.write(tflite_model)
# 使用示例（假设输入是 224x224 RGB 图像）
import numpy as np
dummy_input = np.random.rand(1, 224, 224, 3)  # 替换为你的实际输入
output = keras_model(dummy_input)
print(output)