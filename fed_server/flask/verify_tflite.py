import numpy as np
import tensorflow as tf

def verify_tflite(model_path, input_shape=(1, 5)):
    print("=" * 60)
    print(f"🔍 Verifying model: {model_path}")

    # 读取 TFLite 模型
    with open(model_path, "rb") as f:
        model_buf = f.read()

    interpreter = tf.lite.Interpreter(model_content=model_buf)
    interpreter.allocate_tensors()

    # 打印输入信息
    input_details = interpreter.get_input_details()
    print("Input details:")
    for d in input_details:
        print(f"  name={d['name']}, shape={d['shape']}, dtype={d['dtype']}")

    # 打印输出信息
    output_details = interpreter.get_output_details()
    print("Output details:")
    for d in output_details:
        print(f"  name={d['name']}, shape={d['shape']}, dtype={d['dtype']}")

    # 构造测试输入
    dummy_input = np.random.rand(*input_shape).astype(np.float32)
    interpreter.set_tensor(input_details[0]['index'], dummy_input)
    interpreter.invoke()

    # 打印输出结果
    for i, d in enumerate(output_details):
        output_data = interpreter.get_tensor(d['index'])
        print(f"Output[{i}] → shape={output_data.shape}, values={output_data.flatten()[:8]}")

    print("=" * 60)


if __name__ == "__main__":
    # 修改成你的实际路径
    verify_tflite("ppo_model/actor_task0.tflite", input_shape=(1, 5))
    verify_tflite("ppo_model/critic_task0.tflite", input_shape=(1, 5))
