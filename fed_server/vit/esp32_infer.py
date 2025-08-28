import numpy as np
import tensorflow as tf

# 加载 TFLite 模型
interpreter = tf.lite.Interpreter(model_path="classifier_head.tflite")
interpreter.allocate_tensors()

# 获取输入输出张量信息
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("Input details:", input_details)
print("Output details:", output_details)

# 构造示例输入：假设输入是一个特征向量，比如64维float
# 你需要替换成真实输入数据
input_shape = input_details[0]['shape']
print("Expected input shape:", input_shape)

# 假设输入是float32，shape可能是 (1, 64)
input_data = np.random.random(input_shape).astype(input_details[0]['dtype'])

# 设置输入张量
interpreter.set_tensor(input_details[0]['index'], input_data)

# 执行推理
interpreter.invoke()

# 获取输出
output_data = interpreter.get_tensor(output_details[0]['index'])
print("Output data:", output_data)
