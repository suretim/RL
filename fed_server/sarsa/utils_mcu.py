
import tensorflow as tf
import numpy as np

class TFLiteModelWrapper:
    def __init__(self, tflite_model_path):
        self.interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def predict(self, x):
        # Ensure input is float32
        x = np.array(x, dtype=np.float32)
        self.interpreter.set_tensor(self.input_details[0]['index'], x)
        self.interpreter.invoke()
        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
        return output_data

# ------------------------------------------------------------
# (Optional) Generate trainable tensor indices for ESP32 usage
# ------------------------------------------------------------

def make_indices(model_path: str = "meta_lstm_classifier.tflite", header_path: str = "trainable_tensor_indices.h"):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    tensor_details = interpreter.get_tensor_details()
    trainable_indices = []
    for t in tensor_details:
        name, idx, shape = t['name'], t['index'], t['shape']
        if ("meta_dense" in name or "hvac_dense" in name):
            if "Relu" in name or ";" in name:
                continue
            if len(shape) in (1, 2):
                print(f"Trainable: {name}, index={idx}, shape={shape}")
                trainable_indices.append(idx)
    with open(header_path, "w") as f:
        f.write("#pragma once\n")
        f.write(f"const int trainable_tensor_indices[] = {{{', '.join(map(str, trainable_indices))}}};\n")
        f.write(f"const int trainable_tensor_count = {len(trainable_indices)};\n")
    print("trainable_tensor_indices =", trainable_indices)
    return trainable_indices


