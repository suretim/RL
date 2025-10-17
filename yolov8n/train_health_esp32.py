import yaml
import os
from ultralytics import YOLO
import argparse
import tensorflow as tf
import re
from PIL import Image
import numpy as np



def tflite_to_c_array(tf_file, imgsz, typ):
    """纯Python实现.tflite转C数组"""
    with open(tf_file, 'rb') as f:
        data = f.read()
    
    c_code = f"alignas(16) const unsigned char g_model[] = {{\n"
    for i in range(0, len(data), 12):
        chunk = data[i:i+12]
        hex_str = ", ".join(f"0x{b:02x}" for b in chunk)
        c_code += f"    {hex_str},\n"
    c_code += "};\n"
    c_code += f"const unsigned int MODEL_INPUT_WIDTH = {imgsz};\n"
    c_code += f"const unsigned int MODEL_INPUT_HEIGHT = {imgsz};\n"
    
    with open("model_" + str(imgsz) + "_" + typ + ".h", 'w') as f:
        f.write(c_code)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--imgsz', type=int, default=160)
    args = parser.parse_args()

    # 加载模型配置
    with open(args.data) as f:
        config = yaml.safe_load(f)
    print(f"训练类别: {config['names']}")

    # 初始化YOLOv8模型
    model = YOLO('yolov8n.yaml')

    # 训练配置（适配ESP32硬件限制）
    model.train(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,       # 降低分辨率 160
        device='cpu',           # 强制使用CPU
        augment=False,          # 关闭数据增强
        lr0=0.001,              # 更低学习率
    )
    
    #folder_names = [d for d in os.listdir("runs/detect") if d.startswith("train")]
    #numbers = []
    #for name in folder_names:
    #    match = re.match(r'train(\d*)', name)  # 匹配'train'后可选数字
    #    num = int(match.group(1)) if match and match.group(1) else 0
    #    numbers.append(num)

    # 获取最大值对应的文件夹名
    #max_num = max(numbers)
    target_folder = f"runs/detect/" + f"train{args.imgsz}" if args.imgsz > 0 else "train"
    #target_folder = f"runs/detect/train"

    model = YOLO(target_folder + f"/weights/best.pt")    
    model.export(
        format="tflite",
        #int8=True,                         # 启用 INT8 量化
        int8=False,
        #data="templates/coco8/data.yaml",  # 校准数据集配置文件
        imgsz=args.imgsz,                   # 输入尺寸（需与训练一致）
        half=False,                         # 禁用 FP16（确保全整型）
        keras=False,                        # 禁用 Keras 格式（避免混合精度）
        optimize=False,                     # 关闭自动优化（防止非整型操作插入）
    )

    # 转换为C数组
    #tflite_to_c_array(target_folder + f"/weights/best_saved_model/best_float32.tflite", args.imgsz, "fp32")
    #current_path = os.path.abspath("model_" + str(args.imgsz) + "_fp32.h")
    #print(f"文件已保存到:", current_path)

    # 配置量化参数
    def representative_dataset():
        img_path = f"datasets/raw_images"
        for filename in os.listdir(img_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                img = Image.open(os.path.join(img_path, filename)).convert('RGB')
                resized_img = img.resize((args.imgsz, args.imgsz), Image.BILINEAR)
                img_array = np.array(resized_img)  # 形状为 (H, W, 3)
                img_array = img_array / 255.0 
                img_array = np.expand_dims(img_array, axis=0)
                yield [img_array.astype(np.float32)]  # 转换为float32供量化使用        

    converter = tf.lite.TFLiteConverter.from_saved_model(target_folder + f"/weights/best_saved_model")
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    converter.representative_dataset = representative_dataset
    converter._experimental_disable_per_channel = False
    tflite_quant_model = converter.convert()
    with open(target_folder + f"/weights/best_saved_model/best_int8.tflite", "wb") as f:
        f.write(tflite_quant_model)

    #interpreter = tf.lite.Interpreter(model_path = target_folder + f"/weights/best_saved_model/best_int8.tflite")
    #for op in interpreter.get_tensor_details():
    #    print(op['name'], op['dtype'])  # 确保无float32类型

    # 转换为C数组
    #tflite_to_c_array(target_folder + f"/weights/best_saved_model/best_int8.tflite", args.imgsz, "int8")

    #current_path = os.path.abspath("model_" + str(args.imgsz) + "_int8.h")
    #print(f"文件已保存到:", current_path)

if __name__ == '__main__':
    main()
