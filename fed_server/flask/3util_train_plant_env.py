import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np
from collections import deque
import random
  
import os

from util_agent import PPOBuffer
from util_env import PlantHVACEnv,PlantLLLHVACEnv
from util_agent import  ESP32OnlinePPOFisherAgent
from util_agent import process_experiences,compute_returns,compute_advantages,collect_experiences



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


class SimplePPOAgent:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim

        # 直接构建模型，不编译（因为我们使用自定义训练）
        self.actor = self._build_actor()
        self.critic = self._build_critic()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.0003)

    def _build_actor(self):
        inputs = tf.keras.Input(shape=(self.state_dim,))
        x = tf.keras.layers.Dense(64, activation='relu')(inputs)
        x = tf.keras.layers.Dense(64, activation='relu')(x)
        outputs = tf.keras.layers.Dense(self.action_dim, activation='sigmoid')(x)
        return tf.keras.Model(inputs, outputs)

    def _build_critic(self):
        inputs = tf.keras.Input(shape=(self.state_dim,))
        x = tf.keras.layers.Dense(64, activation='relu')(inputs)
        x = tf.keras.layers.Dense(64, activation='relu')(x)
        outputs = tf.keras.layers.Dense(1, activation='linear')(x)
        return tf.keras.Model(inputs, outputs)

    def select_action(self, state):
        state_tensor = tf.convert_to_tensor([state], dtype=tf.float32)
        action_probs = self.actor(state_tensor, training=False)

        actions = []
        for i in range(self.action_dim):
            prob = action_probs[0, i].numpy()
            actions.append(1 if np.random.random() < prob else 0)

        return np.array(actions), action_probs.numpy()[0]

    def get_value(self, state):
        state_tensor = tf.convert_to_tensor([state], dtype=tf.float32)
        return self.critic(state_tensor, training=False).numpy()[0, 0]

    def train_step(self, batch_data):
        """修改为接受单个批处理数据参数"""
        states, actions, advantages, old_probs, returns = batch_data

        with tf.GradientTape() as tape:
            # Actor前向传播
            new_probs = self.actor(states, training=True)
            new_values = self.critic(states, training=True)

            # 计算损失
            policy_loss, value_loss, entropy = self._compute_loss(
                new_probs, new_values, actions, advantages, old_probs, returns
            )

            total_loss = policy_loss + 0.5 * value_loss - 0.01 * entropy

        grads = tape.gradient(total_loss, self.actor.trainable_variables + self.critic.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.actor.trainable_variables + self.critic.trainable_variables))

        return total_loss

    def _compute_loss(self, new_probs, new_values, actions, advantages, old_probs, returns):
        actions_float = tf.cast(actions, tf.float32)

        # 概率比
        old_action_probs = tf.reduce_sum(old_probs * actions_float + (1 - old_probs) * (1 - actions_float), axis=1)
        new_action_probs = tf.reduce_sum(new_probs * actions_float + (1 - new_probs) * (1 - actions_float), axis=1)

        ratio = new_action_probs / (old_action_probs + 1e-8)
        clipped_ratio = tf.clip_by_value(ratio, 0.8, 1.2)

        # 策略损失
        policy_loss = -tf.reduce_mean(tf.minimum(ratio * advantages, clipped_ratio * advantages))

        # 价值损失
        value_loss = tf.reduce_mean(tf.square(returns - tf.squeeze(new_values)))

        # 熵
        entropy = -tf.reduce_mean(new_probs * tf.math.log(new_probs + 1e-8))

        return policy_loss, value_loss, entropy


# 或者使用@tf.function的版本
class TFFunctionPPOAgent(SimplePPOAgent):
    def __init__(self, state_dim, action_dim):
        super().__init__(state_dim, action_dim)
        # 编译tf.function
        self.train_step_tf = tf.function(self._train_step_impl)

    def _train_step_impl(self, states, actions, advantages, old_probs, returns):
        """tf.function的具体实现"""
        with tf.GradientTape() as tape:
            new_probs = self.actor(states, training=True)
            new_values = self.critic(states, training=True)

            policy_loss, value_loss, entropy = self._compute_loss(
                new_probs, new_values, actions, advantages, old_probs, returns
            )

            total_loss = policy_loss + 0.5 * value_loss - 0.01 * entropy

        grads = tape.gradient(total_loss, self.actor.trainable_variables + self.critic.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.actor.trainable_variables + self.critic.trainable_variables))

        return total_loss

    def train_step(self, batch_data):
        """包装方法，正确传递参数"""
        states, actions, advantages, old_probs, returns = batch_data
        return self.train_step_tf(states, actions, advantages, old_probs, returns)


# 修改训练循环
def train_ppo_simple():
    """简化训练版本：每个episode结束后训练"""
    env = PlantHVACEnv()
    agent = SimplePPOAgent(state_dim=3, action_dim=4)  # 使用非tf.function版本
    buffer = PPOBuffer(state_dim=3, action_dim=4, buffer_size=1024)

    params = {
        "energy_penalty": 0.1,
        "switch_penalty_per_toggle": 0.2,
        "vpd_target": 1.2,
        "vpd_penalty": 2.0
    }

    for episode in range(1000):
        state = env.reset()
        done = False
        episode_reward = 0

        while not done:
            action, action_probs = agent.select_action(state)
            value = agent.get_value(state)

            next_state, reward, done, info = env.step(action, params)

            # 存储经验
            buffer.store(state, action, action_probs, reward, done, value)

            state = next_state
            episode_reward += reward

        # Episode结束后训练
        if buffer.has_enough_samples(32):  # 至少有32个样本
            last_value = agent.get_value(state) if not done else 0
            buffer.finish_path(last_value)

            # 获取所有数据训练
            batch_data = buffer.get_all_data()
            if batch_data is not None:
                # 训练几个epoch
                for epoch in range(3):
                    loss = agent.train_step(batch_data)  # 传递单个参数

                print(f"Episode {episode}, PPO Loss: {loss.numpy():.4f}")

        buffer.clear()
        print(f"Episode {episode}, Reward: {episode_reward:.2f}")


# 批量训练版本
def train_ppo_with_batching():
    env = PlantHVACEnv()
    agent = SimplePPOAgent(state_dim=5, action_dim=4)
    buffer = PPOBuffer(state_dim=5, action_dim=4, buffer_size=1024)

    params = {
        "energy_penalty": 0.1,
        "switch_penalty_per_toggle": 0.2,
        "vpd_target": 1.2,
        "vpd_penalty": 2.0
    }

    batch_size = 32

    for episode in range(1000):
        state = env.reset()
        done = False
        episode_reward = 0

        while not done:
            action, action_probs = agent.select_action(state)
            value = agent.get_value(state)

            next_state, reward, done, info = env.step(action, params)

            buffer.store(state, action, action_probs, reward, done, value)
            state = next_state
            episode_reward += reward

            # 定期训练
            if buffer.has_enough_samples(batch_size):
                last_value = agent.get_value(state) if not done else 0
                buffer.finish_path(last_value)

                # 使用小批量训练
                batch_data = buffer.get_batch(batch_size=batch_size)
                if batch_data is not None:
                    loss = agent.train_step(batch_data)
                    print(f"Step training, Loss: {loss.numpy():.4f}")

                buffer.clear()

        print(f"Episode {episode}, Reward: {episode_reward:.2f}")


# 使用示例
def trainbytask_lifelong_ppo(agent):
    # 创建环境和智能体
    #env = PlantHVACEnv(mode="flowering")
    # 创建多个不同的环境（代表不同任务）
    # PlantLLLHVACEnv states[health, temp, humid, light, co2]
    tasks = [
        PlantLLLHVACEnv(mode="flowering"),
        PlantLLLHVACEnv(mode="seeding"),
        PlantLLLHVACEnv(mode="growing"),
    ]


    # 按顺序学习每个任务
    for task_id, env in enumerate(tasks):
        print(f"开始学习任务 {task_id}...")
        # 收集经验
        experiences = collect_experiences(agent, env, num_episodes=10)

        # 分析收集到的经验
        print(f"总共收集了 {len(experiences)} 条经验")
        states, actions, advantages, old_probs, returns = process_experiences(agent,experiences)

        # 训练多个epoch
        for epoch in range(100):
            loss = agent.train_ppo_step(states=states, actions=actions, advantages=advantages, old_probs=old_probs, returns=returns, use_ewc=True)

        # 保存当前任务知识
        agent.save_task_knowledge((states, actions, advantages, old_probs, returns))

        # 测试所有已学任务的性能（检查是否遗忘）
        for test_id in range(task_id + 1):
            performance = agent.test_task_performance(tasks[test_id])
            print(f"任务 {test_id} 测试性能: {performance}")

        # 回放之前任务的经验
        for _ in range(5):
            agent.replay_previous_tasks(batch_size=32)
        # 计算统计信息
        rewards = [exp['reward'] for exp in experiences]
        health_statuses = [exp['info']['health_status'] for exp in experiences]

        print(f"平均奖励: {np.mean(rewards):.3f}")
        print(f"健康比例: {np.mean([1 if s == 0 else 0 for s in health_statuses]) * 100:.1f}%")
        print(f"不健康比例: {np.mean([1 if s == 1 else 0 for s in health_statuses]) * 100:.1f}%")
 


# 修改训练循环使用SimplePPOAgent
def train_ppo_with_lll(state_dim=5, action_dim=4,):
    # 初始化环境和智能体
    env = PlantLLLHVACEnv()
    #agent = SimplePPOAgent(state_dim=3, action_dim=4)  # 使用简化版本
    agent=ESP32OnlinePPOFisherAgent( state_dim=state_dim, action_dim=action_dim)
    buffer = PPOBuffer(state_dim=state_dim, action_dim=action_dim, buffer_size=2048)
    batch_size=30
    params = {
        "energy_penalty": 0.1,
        "switch_penalty_per_toggle": 0.2,
        "vpd_target": 1.2,
        "vpd_penalty": 2.0
    }

    # 在收集经验时，不要立即训练，而是先存储
    experiences = []

    for episode in range(100):
        state = env.reset()
        done = False
        total_loss=0
        while not done:
            action, old_prob = agent.select_action(state)
            #next_state, reward, done, _ = env.step(action)
            next_state, reward, done, info = env.step(action, params)
            # 存储经验
            experiences.append((state, action, reward,  next_state,done,old_prob))
            state = next_state
        # 每隔一段时间或积累足够经验后，进行批量训练
        if len(experiences) >= batch_size:
            states, actions, advantages, old_probs, returns=process_experiences(agent, experiences, gamma=0.99, gae_lambda=0.95)
            total_loss+=agent.train_step(states, actions, advantages, old_probs, returns)
            # 清空经验缓冲区
            experiences = []


def buffer_train_ppo_with_lll(env=None,agent=None,state_dim=5, action_dim=4):
    # 初始化环境和智能体
    if env is None:
        env = PlantLLLHVACEnv()
    # agent = SimplePPOAgent(state_dim=3, action_dim=4)  # 使用简化版本
    if agent is None:
        agent = ESP32OnlinePPOFisherAgent(state_dim=state_dim, action_dim=action_dim)

    params = {
        "energy_penalty": 0.1,
        "switch_penalty_per_toggle": 0.2,
        "vpd_target": 1.2,
        "vpd_penalty": 2.0
    }

    episode_reward = 0
    # 在收集经验时，不要立即训练，而是先存储
    experiences = []
    buffer_size = 512
    batch_size = 64
    epochs = 10
    state_dim = 5
    action_dim = 4
    buffer = PPOBuffer(state_dim=state_dim, action_dim=action_dim, buffer_size=buffer_size)


    # 假设 buffer 已存满
    for epoch in range(epochs):
        for states_batch, actions_batch, old_probs_batch, returns_batch, values_batch in buffer.get_batch(batch_size):
            states_batch = tf.convert_to_tensor(states_batch, dtype=tf.float32)
            old_probs_batch = tf.convert_to_tensor(old_probs_batch, dtype=tf.float32)

            # actions 做 one-hot
            actions_one_hot = tf.one_hot(actions_batch, depth=action_dim, dtype=tf.float32)

            # Advantage = returns - critic_value
            advantages = tf.convert_to_tensor(returns_batch, dtype=tf.float32) - tf.convert_to_tensor(values_batch,
                                                                                                      dtype=tf.float32)

            # Expand to (batch_size, action_dim)
            advantages = tf.tile(tf.expand_dims(advantages, axis=-1), [1, action_dim])
            returns_expanded = tf.tile(tf.expand_dims(tf.convert_to_tensor(returns_batch, dtype=tf.float32), axis=-1),
                                       [1, action_dim])

            total_loss, policy_loss, value_loss, entropy, ewc_loss = agent.train_step(
                states_batch, actions_one_hot, advantages, old_probs_batch, returns_expanded
            )
            print(f"Epoch {epoch + 1}/{epochs} done, total_loss={total_loss.numpy():.4f}")

    # 清空 buffer
    buffer.clear()


# 或者使用Keras的train_on_batch方法
class CompiledPPOAgent:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.actor, self.critic = self._build_and_compile_networks()

    def _build_and_compile_networks(self):
        # Actor
        actor = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(self.state_dim,)),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(self.action_dim, activation='sigmoid')
        ])
        actor.compile(optimizer='adam', loss='mse')

        # Critic
        critic = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(self.state_dim,)),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(1, activation='linear')
        ])
        critic.compile(optimizer='adam', loss='mse')

        return actor, critic

    def select_action(self, state):
        state_tensor = tf.convert_to_tensor([state], dtype=tf.float32)
        action_probs = self.actor.predict(state_tensor, verbose=0)[0]

        actions = [1 if np.random.random() < prob else 0 for prob in action_probs]
        return np.array(actions), action_probs

    def get_value(self, state):
        state_tensor = tf.convert_to_tensor([state], dtype=tf.float32)
        return self.critic.predict(state_tensor, verbose=0)[0, 0]



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


# 使用示例
if __name__ == "__main__":
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

    agent = ESP32OnlinePPOFisherAgent(state_dim=5, action_dim=4, hidden_units=8)
    #agent = LifelongPPOAgent(state_dim=5, action_dim=4)
    trainbytask_lifelong_ppo( agent=agent)
    #buffer_train_ppo_with_lll(agent=agent)
    # 顯示模型信息
    print("模型參數數量:")
    print(f"Actor: {agent._count_params(agent.actor)}")
    print(f"Critic: {agent._count_params(agent.critic)}")

    # 導出ESP32所需文件
    #agent.export_for_esp32()
    #agent.save_policy_model("ppo_policy", model_type="actor")
    #agent.save_policy_model("ppo_policy", model_type="critic")
    agent.save_policy_model_savedmodel(agent.actor, "ppo_policy", model_type="actor")
    agent.save_policy_model_savedmodel(agent.critic, "ppo_policy", model_type="critic")
    # 測試推理
    test_state = np.array([0, 25.0, 0.5, 500.0, 600.0], dtype=np.float32)
    action = agent.get_action(test_state)
    value = agent.get_value (test_state)

    print(f"測試狀態: {test_state}")
    print(f"預測動作: {action}")
    print(f"預測價值: {value}")

    # 使用最简单的版本避免tf.function问题
    #train_ppo_simple()
    # 配置GPU

    # 使用简化版本的智能体
    #train_ppo_with_lll()
    #trainbytask_lifelong_ppo()

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

