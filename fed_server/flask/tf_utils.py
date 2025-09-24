# tf_utils.py
import tensorflow as tf
import numpy as np
from util_agent import ESP32OnlinePPOFisherAgent
from util_exporter import TensorFlowESP32Exporter
 

def create_representative_dataset(env, policy_net, num_samples=1000, steps_per_episode=10):
    """创建代表性数据集，基于当前状态选择动作"""
    representative_data = []

    for _ in range(num_samples):
        obs = env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]  # 提取状态值

        # 运行环境，采集多步状态
        for step in range(steps_per_episode):
            # 确保obs是正确形状
            obs_reshaped = obs.reshape(1, -1) if len(obs.shape) == 1 else obs
            
            # 根据当前的状态选择动作
            action = policy_net.predict(obs_reshaped)
            if hasattr(action, '__len__') and len(action) > 0:
                action = action[0]  # 取第一个动作
            
            # 执行动作，获取新的状态
            next_obs, _, done, _ = env.step(action)
            if isinstance(next_obs, tuple):
                next_obs = next_obs[0]
            
            representative_data.append(obs.astype(np.float32))
            obs = next_obs
            
            if done:
                break

    return np.array(representative_data)

def create_observations_from_modelpredict(model, env, num_samples=100):
    """
    从环境生成器创建代表性数据集 - 简化版本
    """
    observations = []
    
    obs = env.reset()
    if isinstance(obs, tuple):
        obs = obs[0]

    for i in range(num_samples):
        observations.append(obs.astype(np.float32))
        
        # 预测动作
        obs_reshaped = obs.reshape(1, -1)
        action = model.predict(obs_reshaped)
        if hasattr(action, '__len__') and len(action) > 0:
            action = action[0]
        
        # 执行动作
        obs, _, done, _ = env.step(action)
        if isinstance(obs, tuple):
            obs = obs[0]
        
        if done:
            obs = env.reset()
            if isinstance(obs, tuple):
                obs = obs[0]

    return np.array(observations)


def analyze_tflite_model(model_path):
    """
    分析TFLite模型信息
    """
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print("Input details:", input_details)
    print("Output details:", output_details)

    return input_details, output_details