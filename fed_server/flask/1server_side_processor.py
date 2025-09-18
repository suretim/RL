# server_side_processor_tf.py
import tensorflow as tf
from tensorflow import keras
import numpy as np
import zlib
import sys
import json
import os
import base64
from datetime import datetime
import tensorflow_model_optimization as tfmot
from typing import Dict, Any, Optional, Union

# 你專案內的類（保持不動，假設存在）
from util_exporter import TensorFlowESP32Exporter
from util_hvac_agent import ESP32OnlinePPOFisherAgent,ESP32PPOFisherAgent
from util_env import PlantLLLHVACEnv

MODEL_DIR = "./models"

def ewc_update(actor, fisher, optimal_params, current_params, learning_rate=1e-3, ewc_lambda=500):
    """
    使用 Fisher 矩陣更新 Actor 參數
    actor: tf.keras.Sequential
    fisher: dict
    optimal_params: dict
    current_params: dict, 可用 actor.trainable_variables
    """
    with tf.GradientTape() as tape:
        loss = 0.0
        for var in actor.trainable_variables:
            name = var.name
            diff = var - optimal_params[name]
            loss += tf.reduce_sum(fisher[name] * diff ** 2)
        loss *= ewc_lambda

    grads = tape.gradient(loss, actor.trainable_variables)
    for var, g in zip(actor.trainable_variables, grads):
        if g is not None:
            var.assign_sub(learning_rate * g)

import tensorflow_probability as tfp
import os

tfd = tfp.distributions


class xESP32PPOAgent:
    """專為ESP32設計的輕量級PPO代理"""

    def __init__(self, state_dim=5, action_dim=4,
                 clip_epsilon=0.2, value_coef=0.5, entropy_coef=0.01):
        # 這裡簡單建一個 policy (MLP)
        inputs = keras.Input(shape=(state_dim,))
        x = keras.layers.Dense(32, activation="relu")(inputs)
        outputs = keras.layers.Dense(action_dim, activation="softmax")(x)
        self.policy = keras.Model(inputs, outputs)

        # 其它 PPO 超參數
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef

    def act(self, state):
        """輸入 state -> 輸出 action"""
        state = tf.convert_to_tensor([state], dtype=tf.float32)
        probs = self.policy(state)
        action = tf.random.categorical(tf.math.log(probs), num_samples=1)
        return int(action[0, 0])

    def save_model(self, path="trained_policy_tf.keras"):
        """保存 policy 模型"""
        self.policy.save(path)
        print(f" 模型已保存到 {path}")

    def load_model(self, path="trained_policy_tf.keras"):
        """載入 policy 模型"""
        self.policy = keras.models.load_model(path)
        print(f" 模型已從 {path} 載入")

def generate_smart_representative_data(env, num_samples=1000, mode_weights=None, return_labels=False):
    """
    智能生成代表性数据，覆盖不同模式和工况

    Args:
        env: PlantLLLHVACEnv实例
        num_samples: 总样本数
        mode_weights: 不同模式的权重 [growing, flowering, seeding]
        return_labels: 是否同时返回标签 (X, y)
    """
    if mode_weights is None:
        mode_weights = [0.4, 0.3, 0.3]  # 默认权重

    all_data = []
    all_labels = []

    # 为每个模式生成数据
    modes = ["growing", "flowering", "seeding"]

    for mode, weight in zip(modes, mode_weights):
        num_mode_samples = int(num_samples * weight)
        print(f"为模式 '{mode}' 生成 {num_mode_samples} 个样本")

        # 设置环境模式
        env.mode = mode
        env.reset()

        # 生成该模式的数据
        for i in range(num_mode_samples):
            if mode == "growing":
                action = np.random.choice([0, 1], size=4, p=[0.3, 0.7])
            elif mode == "flowering":
                action = np.random.choice([0, 1], size=4, p=[0.5, 0.5])
            else:  # seeding
                action = np.random.choice([0, 1], size=4, p=[0.7, 0.3])

            # 执行动作
            true_label = modes.index(mode)
            next_state, reward, done, info = env.step(action, true_label=true_label)

            # 添加数据点 & 标签
            current_data_point = env.current_sequence[0, -1]
            all_data.append(current_data_point)
            all_labels.append(true_label)

            if done:
                env.reset()

    # 转 numpy
    all_data = np.array(all_data, dtype=np.float32)
    all_labels = np.array(all_labels, dtype=np.int32)

    # 随机打乱
    idx = np.arange(len(all_data))
    np.random.shuffle(idx)
    all_data, all_labels = all_data[idx], all_labels[idx]

    if return_labels:
        return all_data[:num_samples], all_labels[:num_samples]
    else:
        return all_data[:num_samples]
from util_trainer import LLLTrainer

def env_pipe_trainer(lll_model=None,num_tasks=3,latent_dim=64,num_classes=3,num_epochs_per_task=3,batch_size=32, learning_rate=0.001, ewc_lambda=0.4):
    #env_lll_model, state_dim, action_dim, hidden_units,learning_rate=0.001, ewc_lambda=0.4

    trainer = LLLTrainer(lll_model=lll_model, learning_rate=0.001, ewc_lambda=0.4)

    # 模拟多任务数据
    for task_id in range(num_tasks):
        print(f"\n=== Training Task {task_id + 1} ===")
        num_samples = 200
        latent_features = np.random.randn(num_samples, latent_dim).astype(np.float32)
        labels = np.random.randint(0, num_classes, size=(num_samples,)).astype(np.int32)
        loss=0
        # 按 batch 训练当前任务
        for epoch in range(num_epochs_per_task):
            indices = np.random.permutation(num_samples)
            latent_features_shuffled = latent_features[indices]
            labels_shuffled = labels[indices]

            for start_idx in range(0, num_samples, batch_size):
                end_idx = start_idx + batch_size
                batch_latent = latent_features_shuffled[start_idx:end_idx]
                batch_labels = labels_shuffled[start_idx:end_idx]
                loss = trainer._train_lll_model(batch_latent, batch_labels)
            print(f"  Epoch {epoch + 1}, Last batch loss: {loss:.4f}")

        # 训练完当前任务后，更新 EWC 信息
        trainer.update_ewc(latent_features, labels)


def verify_tflite_model(file_path):
    """正确的TFLite模型验证函数"""
    try:
        with open(file_path, 'rb') as f:
            model_data = f.read()

        if len(model_data) < 16:
            print(" 文件太小")
            return False

        print(f"📊 文件大小: {len(model_data)} 字节")
        print(f"📊 前16字节: {model_data[:16].hex(' ')}")

        # 正确的TFLite魔术数字检测（从第5字节开始）
        if len(model_data) >= 8:
            if model_data[4:8] == b'TFL3':
                print(" 检测到有效的TFLite魔术数字 (TFL3)")
                magic_ok = True
            else:
                print(f" 无效的TFLite魔术数字: {model_data[4:8].hex(' ')}")
                magic_ok = False
        else:
            magic_ok = False

        # 用TensorFlow验证（这是最可靠的）
        try:
            interpreter = tf.lite.Interpreter(model_content=model_data)
            interpreter.allocate_tensors()
            print(" TensorFlow验证成功")
            tf_ok = True
        except Exception as e:
            print(f" TensorFlow验证失败: {e}")
            tf_ok = False

        # 最终结果：只要TensorFlow验证成功就认为有效
        if tf_ok:
            if not magic_ok:
                print("️  注意：文件结构特殊，但TensorFlow可以加载")
            return True
        else:
            return False

    except Exception as e:
        print(f" 文件读取失败: {e}")
        return False

def debug_model_creation(representative_data):
    print("🔍 开始详细调试模型创建过程...")

    # 1. 创建OTA包
    ota_package = exporter.create_ota_package(representative_data, quantize=True)
    print(f"✅ OTA包创建成功")
    print(f"OTA包键值: {list(ota_package.keys())}")

    # 2. 检查model_data_b64
    model_data_b64 = ota_package['model_data_b64']
    print(f"Base64数据长度: {len(model_data_b64)} 字符")
    print(f"Base64前50字符: {repr(model_data_b64[:50])}")

    # 3. Base64解码
    try:
        raw_bytes = base64.b64decode(model_data_b64)
        print(f"✅ Base64解码成功")
        print(f"解码后大小: {len(raw_bytes)} 字节")
        print(f"前32字节: {raw_bytes[:32].hex(' ')}")
    except Exception as e:
        print(f"❌ Base64解码失败: {e}")
        return False

    # 4. 分析数据格式
    print("\n🔍 数据分析:")

    # 检查是否是TFLite格式 - 修正后的逻辑
    if len(raw_bytes) >= 8:  # 需要至少8字节才能检查真正的魔术数字
        # 真正的TFLite魔术数字在第5-8字节（索引4-7）
        magic = raw_bytes[4:8]
        if magic == b'TFL3':
            print("✅ 检测到有效的TFLite魔术数字 (TFL3)")
            print(f"魔术数字位置: 字节4-7: {magic.hex(' ')}")
        else:
            print(f"❌ 不是TFLite格式: 字节4-7 = {magic.hex(' ')}")
            print(f"期望: 54 46 4C 33 (TFL3)")

        # 显示完整的前16字节用于调试
        print(f"前16字节完整数据: {raw_bytes[:16].hex(' ')}")
        print(f"字节0-3: {raw_bytes[:4].hex(' ')} (FlatBuffer头)")
        print(f"字节4-7: {raw_bytes[4:8].hex(' ')} (TFLite魔术数字)")
        print(f"字节8-15: {raw_bytes[8:16].hex(' ')} (其他元数据)")
    else:
        print("❌ 数据太短，无法检查TFLite魔术数字")

    # 检查是否是其他格式
    common_formats = {
        b'\x78\x9C': 'Zlib压缩',
        b'\x1F\x8B': 'Gzip压缩',
        b'PK': 'Zip压缩',
        b'{': 'JSON格式',
        b'<': 'XML格式',
    }

    for magic, format_name in common_formats.items():
        if raw_bytes.startswith(magic):
            print(f"⚠️  检测到可能是 {format_name}")

    # 5. 尝试直接验证
    try:
        interpreter = tf.lite.Interpreter(model_content=raw_bytes)
        interpreter.allocate_tensors()
        print("✅ 直接验证成功 - 数据已经是TFLite格式")

        # 保存文件
        path = os.path.join(MODEL_DIR, "esp32_optimized_model.tflite")
        with open(path, 'wb') as f:
            f.write(raw_bytes)
        print(f"✅ 模型保存成功: {path}")
        return True

    except Exception as e:
        print(f"❌ 直接验证失败: {e}")

    # 6. 尝试作为文本解码（可能是错误信息）
    try:
        text_content = raw_bytes.decode('utf-8')
        print(f"⚠️  数据可以解码为文本:")
        print(f"文本内容: {text_content[:200]}...")

        # 如果是JSON，尝试解析
        if text_content.strip().startswith('{'):
            import json
            try:
                json_data = json.loads(text_content)
                print(f"✅ 是JSON格式: {list(json_data.keys())}")
            except:
                print("❌ 不是有效的JSON")

    except UnicodeDecodeError:
        print("⚠️  数据不是文本格式")

    return False


def analyze_model_details(model_bytes):
    """详细分析模型内容"""
    interpreter0 = tf.lite.Interpreter(model_content=model_bytes)
    interpreter0.allocate_tensors()

    print("=== 模型详细信息 ===")
    print(f"TensorFlow Lite版本: {tf.__version__}")

    # 获取所有操作符
    print("\n=== 操作符列表 ===")
    for i, op in enumerate(interpreter0._get_ops_details()):
        print(f"{i}: {op['op_name']} (index: {op['index']})")

    # 输入输出详情
    print("\n=== 输入张量 ===")
    for i, detail in enumerate(interpreter0.get_input_details()):
        print(f"Input {i}: {detail['name']} {detail['shape']} {detail['dtype']}")

    print("\n=== 输出张量 ===")
    for i, detail in enumerate(interpreter0.get_output_details()):
        print(f"Output {i}: {detail['name']} {detail['shape']} {detail['dtype']}")

    # 检查是否包含不支持的操作符
    micro_supported_ops = ['FULLY_CONNECTED', 'SOFTMAX', 'RESHAPE', 'QUANTIZE', 'DEQUANTIZE']
    all_ops = [op['op_name'] for op in interpreter0._get_ops_details()]

    print("\n=== 兼容性检查 ===")
    for op in all_ops:
        if op not in micro_supported_ops:
            print(f"️  可能不支持的操作符: {op}")
        else:
            print(f" 支持的操作符: {op}")



# -----------------------------
# 使用範例（修正成可 load 的完整模型檔）
# -----------------------------
if __name__ == "__main__":
    # 代表性資料（請替換成你的實際資料）
    # 创建环境
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    latent_dim = 64
    action_dim=4
    num_classes = 3
    batch_size = 32
    num_epochs_per_task = 3
    num_tasks = 3
    env = PlantLLLHVACEnv(seq_len=10,   mode="growing")
    env_pipe_trainer(
        lll_model=env.lll_model ,
        num_tasks=num_tasks,
        latent_dim=latent_dim,
        num_classes=num_classes,
        num_epochs_per_task= num_epochs_per_task,
        batch_size=batch_size,
        learning_rate=0.001,
        ewc_lambda=0.4)

    # 生成代表性数据
    #representative_data = generate_smart_representative_data(env, num_samples=1000)
    representative_data, y_train    = generate_smart_representative_data(env, 100, return_labels=True)

    # 确保形状匹配（如果需要序列数据）
    if len(representative_data.shape) == 2:
        representative_data = representative_data.reshape(-1, 1, 5)
    agent = ESP32PPOFisherAgent(state_dim=env.n_features, action_dim=action_dim, hidden_units=8)

    agent.compute_fisher_matrix(representative_data)
    # 保存 Fisher & Optimal Params
    path_npz=os.path.join(MODEL_DIR, "esp32_fisher.npz")
    agent.save_fisher_and_params(path_npz)
    # 保存 TFLite
    #agent.save_tflite_model("esp32_actor.tflite" )
    path_h5 = os.path.join(MODEL_DIR, "esp32ppo_actor.h5")
    agent.actor.save(path_h5)



    policy_agent=ESP32OnlinePPOFisherAgent(fisher_matrix=agent.fisher_matrix,optimal_params=agent.optimal_params)
    path_policy_h5 = os.path.join(MODEL_DIR, "esp32_OnlinePPOFisher.h5")
    policy_agent.actor.save(path_policy_h5)
    policy_agent.actor.summary()
    # 4. 创建导出器并生成OTA包
    exporter = TensorFlowESP32Exporter(path_policy_h5)
    path_policy_json = os.path.join(MODEL_DIR, "esp32_policy.json")

    # 5. 生成并保存OTA包
    exporter.save_ota_package(
        output_path=path_policy_json,
        representative_data=representative_data,
        fine_tune_data=(representative_data, y_train),
        firmware_version="1.0.0",
        prune=True,  # 启用剪枝
        compress=False,
        quantize=False
    )
    #ota_package = exporter.create_ota_package(representative_data, quantize=True)
    ota_package=exporter.ota_package
    compressed_bytes = base64.b64decode(ota_package['model_data_b64'])

    # 正确的解压方式（确保使用zlib解压）
    try:
        decompressed_bytes = zlib.decompress(compressed_bytes)
    except:
        # 如果解压失败，可能数据没有被压缩
        decompressed_bytes = compressed_bytes

    # 验证TFLite模型有效性
    try:
        # 尝试加载模型来验证
        interpreter = tf.lite.Interpreter(model_content=decompressed_bytes)
        interpreter.allocate_tensors()
        print("✓ TFLite模型验证成功")
    except Exception as e:
        print(f"✗ TFLite模型无效: {e}")
        sys.exit(1)
        # 验证TFLite模型
    # 使用
    #analyze_model_details(decompressed_bytes)
    # 获取所有张量详细信息
    tensor_details = interpreter.get_tensor_details()
    print("TFLite 模型完整层信息")
    print("=" * 80)
    print(f"{'索引':<5} {'名称':<25} {'形状':<15} {'数据类型':<12} {'量化信息'}")
    print("=" * 80)

    for tensor in tensor_details:
        print(
            f"{tensor['index']:<5} {tensor['name']:<25} {str(tensor['shape']):<15} {str(tensor['dtype']):<12} {tensor.get('quantization', 'None')}")

    print("TFLite 模型完整层信息")
    print("=" * 80)
    print(f"{'索引':<5} {'名称':<25} {'形状':<15} {'数据类型':<12} {'量化信息'}")
    print("=" * 80)

    for tensor in tensor_details:
        # 正确处理量化信息（可能是元组或None）
        quantization = tensor.get('quantization', ())

        if quantization and isinstance(quantization, (list, tuple)) and len(quantization) >= 2:
            scale, zero_point = quantization[0], quantization[1]
            quant_info = f"scale:{scale}, zero_point:{zero_point}"
        else:
            quant_info = "quantization free"

        print(
            f"{tensor['index']:<5} {str(tensor['name'])[:24]:<25} {str(tensor['shape']):<15} {str(tensor['dtype']):<12} {quant_info}")

    print("=" * 80)
    print(f" 总层数: {len(tensor_details)}")
    print(f" 模型大小: {len(decompressed_bytes):,} 字节")
    path_policy_tflite = os.path.join(MODEL_DIR, "esp32_optimized_model.tflite")
    with open(path_policy_tflite, 'wb') as f:
        f.write(decompressed_bytes)

    print(f"✓ 模型已保存到: {path_policy_tflite}")
    print(f"模型大小: {len(decompressed_bytes)} 字节")

    # 使用Post-Training Quantization进行量化
    #model = tf.keras.models.load_model(path_policy_h5)

    #quantize_model = tf.quantization.quantize(model)

    # 将量化后的模型保存为 TFLite 格式
    #converter = tf.lite.TFLiteConverter.from_keras_model(quantize_model)
    #converter.optimizations = [tf.lite.Optimize.DEFAULT]

    # 设置支持量化的操作
    #converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    #converter.representative_dataset = representative_data
    #tflite_model = converter.convert()
    #with open(path_policy_tflite, 'wb') as f:
    #    f.write(tflite_model)
    # 运行调试
    debug_model_creation(representative_data)
    # 在保存后立即验证
    if verify_tflite_model(path_policy_tflite):
        print("PC端验证通过，可以上传到ESP32")
    else:
        print("PC端验证失败，请检查模型生成过程")
    # 也可單獨呼叫
    #_ = exporter.apply_quantization(representative_data)
    #_ = exporter.compute_fisher_matrix(representative_data)
