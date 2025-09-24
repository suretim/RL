import tensorflow as tf
import numpy as np
import json
import os
from tf_utils import create_representative_dataset
from util_agent import ESP32PPOAgent, ESP32OnlinePPOFisherAgent,PPOBuffer
from util_exporter import TensorFlowESP32Exporter
from util_env import PlantLLLHVACEnv
 
 


def main():
    
    # GPU设置
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    env=PlantLLLHVACEnv()
 
    # 初始化缓冲区
    buf = PPOBuffer(
        state_dim=env.state_dim,      # 状态维度
        action_dim=env.action_dim,    # 动作维度（离散动作空间的大小）
        buffer_size=512,          # 缓冲区大小
        gamma=0.99               # 折扣因子
    )
    try:
        # 尝试使用在线学习代理
        print("开始训练ESP32OnlinePPOFisherAgent...")
        
        # 1. 训练模型
        agent = ESP32OnlinePPOFisherAgent()
        agent.learn(buf,total_timesteps=1000000)

        # 2. 保存原始模型
        # agent.policy.save("trained_policy_tf")

        # 3. 创建导出器
        exporter = TensorFlowESP32Exporter(agent.policy)

        # 4. 生成代表性数据
        
        """从生成器创建代表性数据集"""
        
        # 这里需要确保policy可以用于预测，或者使用agent的policy
        representative_data= create_representative_dataset(env, agent.policy_net, num_samples=1000)

        # 5. 创建OTA包
        exporter.save_ota_base_package(
            "esp32_ota_package.json",
            representative_data,
            firmware_version="1.0.0",
            prune=True,
            quantize=True
        ) 

        print("训练和导出完成!")
        
    except Exception as e:
        print(f"在线训练失败: {e}")
        print("回退到基本ESP32PPOAgent...")
        
        # 回退方案：使用基本代理
        esp32_agent = ESP32PPOAgent(state_dim=5, action_dim=4, hidden_units=8)
        esp32_agent.count_agent_params()
        # 训练基本代理（需要实现训练逻辑）
        # 这里假设ESP32PPOAgent有训练方法
        if hasattr(esp32_agent, 'learn'):
            esp32_agent.learn(buf,total_timesteps=10000)
        buf.clear()
        # 显示模型信息
        print("模型參數數量:")
        
        # 修复参数计数方法
        def count_params(model):
            """计算模型参数数量"""
            if hasattr(model, 'count_params'):
                return model.count_params()
            elif hasattr(model, 'trainable_variables'):
                return sum([np.prod(v.shape) for v in model.trainable_variables])
            else:
                return "无法计算"
        
        print(f"Actor: {count_params(esp32_agent.actor)}")
        print(f"Critic: {count_params(esp32_agent.critic)}")

        # 导出ESP32所需文件（需要确保方法存在）
        if hasattr(esp32_agent, 'export_for_esp32'):
            esp32_agent.export_for_esp32()
        else:
            # 备用导出方法
             
            exporter = TensorFlowESP32Exporter(esp32_agent)
            representative_data= create_representative_dataset(env, esp32_agent.policy_net, num_samples=1000)
            exporter.save_ota_base_package(
                "esp32_backup_package.json",
                representative_data,
                firmware_version="1.0.0",
                prune=True,
                quantize=True
            )


# 使用示例
if __name__ == "__main__":
    main()