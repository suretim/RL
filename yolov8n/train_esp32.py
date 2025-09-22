import yaml
import os
from ultralytics import YOLO
import argparse
import tensorflow as tf
import re
from PIL import Image
import numpy as np
'''
1.  安装Python 3.10.12
python3 -V
        Python 3.10.12

2.  创建python虚拟环境
python3 -m venv tflm_p310_env

3.  激活虚拟环境
source ./tflm_p310_env/bin/activate

4.  安装依赖库
pip3 install "sng4onnx>=1.0.1" "onnx_graphsurgeon>=0.3.26" "ai-edge-litert>=1.2.0,<1.4.0" "onnx>=1.12.0,<1.18.0" "onnx2tf>=1.26.3" "onnxslim>=0.1.59" "onnxruntime" "ultralytics" "tensorflow" "flask" 

5.   源文件组织结构：
        app.py
        train_esp32.py
        calibration_image_sample_data_20x128x128x3_float32.npy
        templates/index.html
        数据集压缩文件dataset.zip

6.   训练
        python3 app.py
 python3 app.py
 * Serving Flask app 'app'
 * Debug mode: on
WARNING: This is a development server. Do not use it in a production deployment. Use a production WSGI server instead.
 * Running on all addresses (0.0.0.0)
 * Running on http://127.0.0.1:5000
 
         在浏览器上打开http://127.0.0.1:5000，选择数据集，输入目标类别，点击分割数据集；选择输入尺寸，输入训练轮次/批次大小，点击开始训练
         
         查看训练效果：

                   all         58         58       0.99      0.976      0.994      0.854
                    aa         37         37      0.986          1      0.995       0.88
                    bb         21         21      0.994      0.952      0.993      0.828
Speed: 0.0ms preprocess, 3.0ms inference, 0.0ms loss, 0.1ms postprocess per image

        训练输出数组文件：
        文件已保存到: /mnt/c/csz/tokay_lite_sdk/esp32s3cam_yolov8/yolov8_tflm/esp32s3_yolo_project/model_160.h
        
7.  部署
        a. 把上述文件的内容全部复制到camera_http_example项目下的model.h, 编译，烧录
        b. 电脑连接esp32s3cam热点ESP32-CAM-AP，密码12345678
        c. 在浏览器上打开192.168.4.1:81, 在控制命令输入框输入tflite=1,点击发送命令，开始对图像执行目标识别推理
'''

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
    
    folder_names = [d for d in os.listdir("runs/detect") if d.startswith("train")]
    numbers = []
    for name in folder_names:
        match = re.match(r'train(\d*)', name)  # 匹配'train'后可选数字
        num = int(match.group(1)) if match and match.group(1) else 0
        numbers.append(num)

    # 获取最大值对应的文件夹名
    max_num = max(numbers)
    target_folder = f"runs/detect/" + f"train{max_num}" if max_num > 0 else "train"

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
    tflite_to_c_array(target_folder + f"/weights/best_saved_model/best_float32.tflite", args.imgsz, "fp32")
    current_path = os.path.abspath("model_" + str(args.imgsz) + "_fp32.h")
    print(f"文件已保存到:", current_path)

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
    tflite_to_c_array(target_folder + f"/weights/best_saved_model/best_int8.tflite", args.imgsz, "int8")

    current_path = os.path.abspath("model_" + str(args.imgsz) + "_int8.h")
    print(f"文件已保存到:", current_path)

if __name__ == '__main__':
    main()
