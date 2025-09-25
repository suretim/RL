import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np
from collections import deque
import random
  
import os

from util_agent import PPOBuffer
from util_env import PlantLLLHVACEnv
from util_agent import  ESP32OnlinePPOFisherAgent ,PPOAgent


class MemoryBuffer:
    def __init__(self, capacity=1000):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def update(self, X, y):
        if not isinstance(X, tf.Tensor):
            X = tf.convert_to_tensor(X, dtype=tf.float32)
        if not isinstance(y, tf.Tensor):
            y = tf.convert_to_tensor(y, dtype=tf.int32)
        self.buffer.append((X, y))

    def sample(self, batch_size):
        if len(self.buffer) < batch_size:
            return random.sample(list(self.buffer), len(self.buffer))
        return random.sample(list(self.buffer), batch_size)


def build_encoder(seq_len, n_features, latent_dim):
    """构建TensorFlow版本的encoder"""
    inputs = tf.keras.Input(shape=(seq_len, n_features))
    x = layers.LSTM(64, return_sequences=True)(inputs)
    x = layers.LSTM(latent_dim)(x)
    return Model(inputs, x)




class lifelonglearningModel(Model):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(lifelonglearningModel, self).__init__()
        self.dense1 = layers.Dense(hidden_dim, activation='relu')
        self.dense2 = layers.Dense(output_dim, activation='softmax')

        self.previous_weights = {}
        self.fisher_matrices = {}
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    def call(self, x):
        x = self.dense1(x)
        return self.dense2(x)

    def compute_fisher_matrix(self, X, y, num_samples=100):
        fisher_matrix = {}
        for var in self.trainable_variables:
            fisher_matrix[var.name] = tf.zeros_like(var)

        for _ in range(num_samples):
            with tf.GradientTape() as tape:
                output = self(X, training=True)
                loss = tf.keras.losses.sparse_categorical_crossentropy(y, output)

            grads = tape.gradient(loss, self.trainable_variables)
            for var, grad in zip(self.trainable_variables, grads):
                if grad is not None:
                    fisher_matrix[var.name] += tf.square(grad) / num_samples

        return fisher_matrix

    def compute_ewc_loss(self, fisher_matrix, previous_weights, lambda_ewc=1000):
        ewc_loss = 0
        for var in self.trainable_variables:
            if var.name in fisher_matrix and var.name in previous_weights:
                ewc_loss += tf.reduce_sum(
                    fisher_matrix[var.name] * tf.square(var - previous_weights[var.name])
                )
        return lambda_ewc * ewc_loss




def buffer_train_ppo_with_lll():
    params = {
        "energy_penalty": 0.1,
        "switch_penalty_per_toggle": 0.2,
        "vpd_target": 1.2,
        "vpd_penalty": 2.0
    }
    tasks = [
        PlantLLLHVACEnv(mode="flowering"),
        PlantLLLHVACEnv(mode="seeding"),
        PlantLLLHVACEnv(mode="growing"),
    ]

    # 按顺序学习每个任务
    for task_id, env in enumerate(tasks):
        print(f"开始学习任务 {task_id}...")

        state_dim = env.state_dim
        action_dim = env.action_dim
        agent = PPOAgent(state_dim, action_dim)

        # 训练当前任务
        agent.collect_and_train(env=env, num_episodes=100)

        # 如果是最后一个任务之前的任务，保存当前任务的参数用于EWC
        if task_id < len(tasks) - 1:
            # 保存当前任务的重要权重和参数
            agent.save_task_parameters(task_id)

        print(f"任务 {task_id} 学习完成")



def debug_buffer_contents(buffer, batch_size):
    """调试缓冲区内容 - 修复版本"""
    print("=== 缓冲区调试信息 ===")

    # 安全地检查缓冲区大小
    try:
        if hasattr(buffer, '__len__'):
            print(f"缓冲区大小: {len(buffer)}")
        else:
            print("缓冲区没有 __len__ 方法")
    except Exception as e:
        print(f"检查缓冲区大小时出错: {e}")

    # 检查是否准备好
    try:
        is_ready = buffer.is_ready(batch_size)
        print(f"是否准备好: {is_ready}")
    except Exception as e:
        print(f"检查缓冲区准备状态时出错: {e}")
        is_ready = False

    # 检查缓冲区的主要属性
    buffer_attrs = ['states', 'actions', 'rewards', 'dones', 'values', 'log_probs', 'returns', 'advantages']
    for attr in buffer_attrs:
        if hasattr(buffer, attr):
            try:
                data = getattr(buffer, attr)
                if hasattr(data, '__len__'):
                    print(f"{attr} 大小: {len(data)}")
                    if len(data) > 0:
                        sample = data[0] if isinstance(data, list) else data[:1]
                        print(f"  {attr} 样本类型: {type(sample)}")
                else:
                    print(f"{attr} 类型: {type(data)}")
            except Exception as e:
                print(f"检查 {attr} 时出错: {e}")


def process_batch_data(batch_bytes):
    """处理可能的字节数据"""
    try:
        import pickle
        return pickle.loads(batch_bytes)
    except:
        print("无法反序列化批次数据")
        return []


def safe_train_step(agent, **kwargs):
    """安全的训练步骤包装器"""
    try:
        # 检查所有输入参数
        for key, value in kwargs.items():
            if value is None:
                print(f"警告: {key} 为 None")
                return None, 0, 0, 0, 0

        return agent.train_step(**kwargs)
    except Exception as e:
        print(f"train_step错误: {e}")
        import traceback
        traceback.print_exc()
        return None, 0, 0, 0, 0


import pickle


def check_and_reload_actor_model(agent):
    """检查并重新加载actor模型"""
    print("=== 检查Actor模型状态 ===")

    # 检查actor的类型
    print(f"Actor类型: {type(agent.actor)}")

    # 如果actor是字节对象，需要重新加载
    if isinstance(agent.actor, bytes):
        print("检测到actor为字节对象，正在重新加载模型...")
        try:
            # 尝试从字节加载模型
            if hasattr(agent, 'actor_architecture'):
                # 重新创建模型结构
                agent.actor = agent.actor_architecture()
                print("从架构重新创建actor模型")
            else:
                # 尝试反序列化
                agent.actor = tf.keras.models.model_from_json(agent.actor)
                print("从JSON重新加载actor模型")
        except Exception as e:
            print(f"重新加载actor模型失败: {e}")
            # 创建新的模型
            agent.actor = create_new_actor_model(agent.observation_space, agent.action_space)
            print("创建新的actor模型")

    # 确保actor是可调用的
    if not callable(agent.actor):
        print("Actor不可调用，重新初始化...")
        agent.actor = create_new_actor_model(agent.observation_space, agent.action_space)

    print("Actor模型检查完成")


def create_new_actor_model(observation_space, action_space):
    """创建新的actor模型"""
    try:
        from tensorflow.keras import layers, Model

        # 简单的actor网络结构
        inputs = layers.Input(shape=(observation_space,))
        x = layers.Dense(64, activation='relu')(inputs)
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dense(32, activation='relu')(x)
        outputs = layers.Dense(action_space, activation='softmax')(x)

        model = Model(inputs=inputs, outputs=outputs)
        return model
    except Exception as e:
        print(f"创建新actor模型失败: {e}")
        return None


def check_agent_models(agent):
    """检查agent的所有模型状态"""
    print("=== 检查Agent所有模型 ===")

    model_names = ['actor', 'critic', 'optimizer']
    for model_name in model_names:
        if hasattr(agent, model_name):
            model = getattr(agent, model_name)
            print(f"{model_name}: 类型={type(model)}, 可调用={callable(model)}")

            if isinstance(model, bytes):
                print(f"警告: {model_name} 是字节对象!")

    # 特别检查actor
    if hasattr(agent, 'actor'):
        check_and_reload_actor_model(agent)


def safe_select_action(agent, state):
    """安全的动作选择函数"""
    try:
        # 检查actor模型状态
        if not hasattr(agent, 'actor') or agent.actor is None:
            print("Actor模型不存在，创建新模型...")
            agent.actor = create_new_actor_model(len(state), agent.action_space)

        if isinstance(agent.actor, bytes):
            print("Actor是字节对象，重新加载...")
            check_and_reload_actor_model(agent)

        if not callable(agent.actor):
            print("Actor不可调用，使用随机动作...")
            return np.random.randint(agent.action_space), 0.0, 0.0

        # 正常选择动作
        state_tensor = tf.convert_to_tensor([state], dtype=tf.float32)
        action_probs = agent.actor(state_tensor)
        action = tf.random.categorical(tf.math.log(action_probs), 1)[0, 0]
        action_prob = action_probs[0, action]

        # 获取价值估计
        if hasattr(agent, 'critic') and callable(agent.critic):
            value = agent.critic(state_tensor)[0, 0]
        else:
            value = 0.0

        return action.numpy(), action_prob.numpy(), value

    except Exception as e:
        print(f"选择动作时出错: {e}")
        # 返回随机动作作为后备
        return np.random.randint(agent.action_space), 1.0 / agent.action_space, 0.0


def buffer_train_ppo_with_lll_alternative(env, agent, task_id=0):
    batch_size = 10
    epochs = 100

    # 首先检查agent模型状态
    check_agent_models(agent)

    ppo_buffer = agent.ppo_buffer
    state = env.reset()
    episode_reward = 0
    episode_count = 0

    print(f"开始训练任务 {task_id}...")

    while episode_count < epochs:
        try:
            # 使用安全的动作选择
            action, action_prob, value = safe_select_action(agent, state)
            next_state, reward, done, _ = env.step(action)

            # 存储经验到缓冲区
            ppo_buffer.store(state, action, action_prob, reward, next_state, done, value)
            state = next_state
            episode_reward += reward

            if done:
                ppo_buffer.finish_path()
                state = env.reset()
                episode_count += 1
                #print(f"任务 {task_id}, 回合 {episode_count}: 奖励 = {episode_reward}")
                episode_reward = 0

                # 检查缓冲区是否准备好训练
                try:
                    is_ready = ppo_buffer.is_ready(batch_size)
                except Exception as e:
                    print(f"检查缓冲区准备状态时出错: {e}")
                    is_ready = False
                if is_ready and episode_count % 10 == 0:  # 每10回合训练一次
                    try:
                        #print("开始训练...")

                        # 获取训练批次生成器
                        batch_generator = ppo_buffer.get_ppo_batch(batch_size)

                        successful_batches = 0
                        total_batches = 0

                        # 遍历所有mini-batch
                        for batch in batch_generator:
                            total_batches += 1

                            # 正确解包7个值
                            if len(batch) == 7:
                                states_batch, actions_batch, old_probs_batch, returns_batch, \
                                    values_batch, next_states_batch, dones_batch = batch

                                # 执行PPO训练
                                result = agent.train_on_batch(
                                    states_batch, actions_batch, old_probs_batch, returns_batch,
                                    values_batch, next_states_batch, dones_batch
                                )
                                if result and result[0] is not None:
                                    total_loss, policy_loss, value_loss, entropy, ewc_loss = result
                                    #print(f"训练成功 - 总损失: {total_loss:.4f}")
                                    successful_batches += 1

                            else:
                                print(f"批次格式错误: 期望7个值，得到{len(batch)}个")

                        #print(f"成功训练 {successful_batches}/{total_batches} 个批次")

                    except Exception as e:
                        print(f"训练过程中出错: {e}")
                        import traceback
                        traceback.print_exc()

        except Exception as e:
            print(f"回合处理过程中出错: {e}")
            state = env.reset()
            episode_reward = 0
            continue

    print(f"任务 {task_id} 训练完成")
    return episode_count


# 修复agent的序列化问题
def fix_agent_serialization(agent):
    """修复agent的序列化问题"""
    print("=== 修复Agent序列化 ===")

    # 检查并修复所有模型
    model_attrs = ['actor', 'critic', 'optimizer']

    for attr in model_attrs:
        if hasattr(agent, attr):
            current_value = getattr(agent, attr)

            if isinstance(current_value, bytes):
                print(f"修复 {attr} 的序列化问题...")
                try:
                    # 尝试反序列化
                    if attr in ['actor', 'critic']:
                        # 对于Keras模型
                        reconstructed_model = tf.keras.models.model_from_json(
                            current_value.decode('utf-8') if isinstance(current_value, bytes) else current_value
                        )
                        setattr(agent, attr, reconstructed_model)
                        print(f"{attr} 反序列化成功")
                    else:
                        # 对于其他对象
                        reconstructed_obj = pickle.loads(current_value)
                        setattr(agent, attr, reconstructed_obj)
                        print(f"{attr} 反序列化成功")

                except Exception as e:
                    print(f"{attr} 反序列化失败: {e}")
                    # 设置为None，后续会重新创建
                    setattr(agent, attr, None)




class TFLitePPOTrainer:
    def __init__(self):
        self.model = self._build_tflite_compatible_model()

    def _build_tflite_compatible_model(self):
        """构建兼容TFLite的简单模型"""
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(8, activation='relu', input_shape=(3,)),
            tf.keras.layers.Dense(4, activation='sigmoid', name='action_probs')
        ])
        return model

    def train(self, experiences):
        """训练模型"""
        # 简化训练逻辑
        states = np.array([exp['state'] for exp in experiences])
        advantages = np.array([exp['advantage'] for exp in experiences])

        # 这里使用简化训练，实际应该用PPO算法
        self.model.fit(states, advantages, epochs=10, verbose=0)

    def convert_to_tflite(self, output_path):
        """转换为TFLite模型"""
        # 转换模型
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]  # 优化模型大小
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,  # 启用TFLite内置操作
            tf.lite.OpsSet.SELECT_TF_OPS  # 启用TensorFlow操作
        ]

        tflite_model = converter.convert()

        # 保存模型
        with open(output_path, 'wb') as f:
            f.write(tflite_model)

        print(f"Model converted to TFLite: {output_path}")
        print(f"Model size: {len(tflite_model)} bytes")

        return tflite_model

    def generate_c_array(self, tflite_model, output_path):
        """生成C数组格式的模型"""
        with open(output_path, 'w') as f:
            f.write('#ifndef MODEL_DATA_H\n')
            f.write('#define MODEL_DATA_H\n\n')
            f.write('#include <cstdint>\n\n')
            f.write(f'const unsigned char g_model[] = {{\n')

            # 每行16个字节
            for i in range(0, len(tflite_model), 16):
                line = ', '.join(f'0x{byte:02x}' for byte in tflite_model[i:i + 16])
                f.write(f'  {line},\n')

            f.write('};\n')
            f.write(f'const unsigned int g_model_len = {len(tflite_model)};\n')
            f.write('#endif\n')

        print(f"C array header generated: {output_path}")



if __name__ == "__main__":
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

    #agent = ESP32OnlinePPOFisherAgent(state_dim=5, action_dim=4, hidden_units=8)
    #
    #buffer_train_ppo_with_lll()
    params = {
        "energy_penalty": 0.1,
        "switch_penalty_per_toggle": 0.2,
        "vpd_target": 1.2,
        "vpd_penalty": 2.0
    }
    tasks = [
        PlantLLLHVACEnv(mode="flowering"),
        PlantLLLHVACEnv(mode="seeding"),
        PlantLLLHVACEnv(mode="growing"),
    ]
    try:


        for task_id, env in enumerate(tasks):
            state_dim = env.state_dim
            action_dim = env.action_dim
            # 确保正确设置离散/连续参数
            agent = PPOAgent(state_dim, action_dim, is_discrete=True)  # 明确设置为离散

            print(f"开始学习任务 {task_id}...")
            # 在使用agent之前调用修复函数
            fix_agent_serialization(agent)
            # 测试任务性能（安全调用）
            try:
                performance = agent.test_task_performance(tasks[task_id])
                print(f"任务 {task_id} 性能: {performance}")
            except ValueError as e:
                print(f"测试任务性能时出错: {e}")
                # 使用调试版本
                performance = agent.test_task_performance_debug(tasks[task_id])
            except Exception as e:
                print(f"测试任务性能时未知错误: {e}")
                performance = 0.0
            # 使用调试版本
            #test_state = env.reset()
            #print("测试act方法...")
            #action = agent.act_debug(test_state)
            buffer_train_ppo_with_lll_alternative(env=env,agent=agent,task_id=task_id)
            #trainbytask_lifelong_ppo(env=env,agent=agent)
            if task_id < len(tasks) - 1:
                agent.save_task_parameters(task_id)

            print("模型参数数量:")
            print(f"Actor: {agent._count_params(agent.actor)}")
            print(f"Critic: {agent._count_params(agent.critic)}")

            agent.export_for_esp32()
            print(f"任务 {task_id} 学习完成")

    except Exception as e:
        print(f"训练过程中出错: {e}")
    import traceback

    traceback.print_exc()
    '''
    trainer = TFLitePPOTrainer()

    # 模拟训练数据
    dummy_experiences = [
        {'state': [0, 25.0, 0.5], 'advantage': [0.8, 0.2, 0.1, 0.9]},
        {'state': [1, 30.0, 0.8], 'advantage': [0.1, 0.9, 0.2, 0.8]}
    ]

    trainer.train(dummy_experiences)
    tflite_model = trainer.convert_to_tflite('ppo_model.tflite')
    trainer.generate_c_array(tflite_model, 'model_data.h')
    '''

