import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np
import tensorflow_probability as tfp

import datetime
from collections import deque
import os
import json
from util_env import PlantLLLHVACEnv
tfd = tfp.distributions

# state_dim = 5  # 健康狀態、溫度、濕度、光照、CO2
# action_dim = 4  # 4個控制動作
class PPOBaseAgent:
    def __init__(self, fisher_matrix=None, optimal_params=None, state_dim=5, action_dim=4,
                 clip_epsilon=0.2, value_coef=0.5, entropy_coef=0.01,
                 ewc_lambda=500, memory_size=1000, hidden_units=32,
                 clip_ratio=0.2, actor_lr=1e-4, critic_lr=1e-3, is_discrete=False,
                 gamma=0.99, lam=0.95):

        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.is_discrete = is_discrete
        self.hidden_units = hidden_units
        self.ewc_lambda = ewc_lambda  # EWC正则化强度
        self.env=PlantLLLHVACEnv()
        # 构建网络
        self.policy = ESP32PPOPolicy(state_dim, action_dim, hidden_units, is_discrete)

        self.critic = self._build_critic()

        self.value_net = self.critic
        self.actor = self._build_actor()
        self.policy_params = self.actor.trainable_variables
        self.value_params  = self.critic.trainable_variables
        # 持续学习相关
        # 关键修复：强制初始化为字典
        self._initialize_ewc_variables()
        self.current_task_id = 0
        self.log_std = tf.Variable(initial_value=-0.5 * tf.ones(action_dim), trainable=True)

        # 使用更小的學習率
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
        # 初始化经验缓冲区
        self.buffer_size = 10000
        self._init_buffer()
        # EWC相关变量
        self.ewc_task_count = 0
        if fisher_matrix is not None:
            self.fisher_matrix = fisher_matrix
        else:
            self.fisher_matrix = {}
        if optimal_params is not None:
            self.optimal_params = optimal_params
        else:
            self.optimal_params = {}


        self.old_policy_params = None
        self.ewc_lambda = ewc_lambda
        self.hidden_units = hidden_units
        # 关键修复：确保正确初始化持续学习相关变量
        self.task_memory = deque(maxlen=memory_size)


        self.ewc_means = {}  # 保存每个任务的参数均值
        self.ewc_fisher = {}  # 保存每个任务的Fisher信息矩阵
        self.ppo_buffer=PPOBuffer(self.state_dim,self.action_dim)
        self.action_space = type('', (), {})()  # 生成一个假的对象
        self.action_space.n = action_dim
    def _initialize_ewc_variables(self):
        """强制初始化EWC相关变量为正确的类型"""
        # 删除可能存在的错误变量
        for attr_name in ['optimal_params', 'fisher_matrices', 'ewc_means', 'ewc_fisher', 'task_memory']:
            if hasattr(self, attr_name):
                delattr(self, attr_name)

        # 重新初始化为正确类型
        self.optimal_params = {}
        self.fisher_matrices = {}
        self.ewc_means = {}
        self.ewc_fisher = {}
        self.task_memory = deque(maxlen=1000)  # 使用固定大小

        print("EWC变量已重新初始化")
    def _init_buffer(self):
        """初始化经验缓冲区"""
        self.states = np.zeros((self.buffer_size, self.state_dim))
        self.actions = np.zeros(self.buffer_size, dtype=np.int32)
        self.probs = np.zeros((self.buffer_size, self.action_dim))
        self.rewards = np.zeros(self.buffer_size)
        self.dones = np.zeros(self.buffer_size, dtype=bool)
        self.pointer = 0
        self.buffer_full = False

    def _build_actor(self):
        return tf.keras.Sequential([
            tf.keras.layers.Dense(self.hidden_units, activation='relu',
                                  input_shape=(self.state_dim,)),
            tf.keras.layers.Dense(self.hidden_units, activation='relu'),  # 增加一层
            tf.keras.layers.Dense(self.action_dim, activation='tanh')  # 改为tanh，输出范围[-1,1]
        ], name='esp32_actor')

    def _build_critic(self):
        return tf.keras.Sequential([
            tf.keras.layers.Dense(self.hidden_units, activation='relu',
                                  input_shape=(self.state_dim,)),
            tf.keras.layers.Dense(self.hidden_units, activation='relu'),  # 增加一层
            tf.keras.layers.Dense(1, activation='linear')
        ], name='esp32_critic')

    def get_action(self, state):
        """根据策略网络选择动作 - 安全版本"""
        try:
            state = np.expand_dims(state, axis=0).astype(np.float32)

            if self.is_discrete:
                logits = self.policy(state)
                dist = tfp.distributions.Categorical(logits=logits)
                action = dist.sample()[0].numpy()
                action_prob = dist.prob(action).numpy()
                return int(action), float(action_prob)
            else:
                mean = self.policy(state)
                log_std = tf.ones_like(mean) * self.log_std
                dist = tfp.distributions.Normal(loc=mean, scale=tf.exp(log_std))
                action = dist.sample()[0].numpy()
                action_prob = dist.prob(action).numpy()
                action_prob = np.prod(action_prob)
                return action, float(action_prob)

        except Exception as e:
            print(f"获取动作时出错: {e}")
            if self.is_discrete:
                return 0, 0.25
            else:
                return np.zeros(self.action_dim), 0.5

    def get_value(self, state):
        state_tensor = np.expand_dims(state, axis=0)  # (1, state_dim)
        value = self.critic(state_tensor).numpy()[0, 0]
        return value

    def _count_params(self, model):
        """計算模型參數數量"""
        return sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])

    def _compute_fisher_matrix(self, experiences):
        """计算Fisher矩阵 - 彻底修复版本"""
        try:
            states, actions, _, old_probs, _ = experiences

            print(f"Fisher计算输入形状: states={states.shape}, actions={actions.shape}, old_probs={old_probs.shape}")

            # 确保输入是张量，正确处理数据类型
            states = tf.convert_to_tensor(states, dtype=tf.float32)

            # 关键修复：动作应该是int32（离散动作索引），不是float32
            if self.is_discrete:
                actions = tf.convert_to_tensor(actions, dtype=tf.int32)
            else:
                actions = tf.convert_to_tensor(actions, dtype=tf.float32)

            old_probs = tf.convert_to_tensor(old_probs, dtype=tf.float32)

            # 获取所有可训练变量
            trainable_vars = self.actor.trainable_variables + self.critic.trainable_variables
            fisher_accum = [tf.zeros_like(var, dtype=tf.float32) for var in trainable_vars]

            batch_size = tf.shape(states)[0]
            print(f"计算Fisher矩阵，批次大小: {batch_size}")

            # 使用整个批次计算（简化版本）
            with tf.GradientTape() as tape:
                if self.is_discrete:
                    # 离散动作空间：使用分类分布
                    logits = self.actor(states, training=False)
                    dist = tfp.distributions.Categorical(logits=logits)

                    # 计算所选动作的对数概率
                    log_probs = dist.log_prob(actions)  # [batch_size]
                else:
                    # 连续动作空间：使用正态分布
                    mean = self.actor(states, training=False)
                    log_std = tf.ones_like(mean) * self.log_std
                    dist = tfp.distributions.Normal(loc=mean, scale=tf.exp(log_std))
                    log_probs = tf.reduce_sum(dist.log_prob(actions), axis=-1)  # [batch_size]

                # 计算平均对数概率
                logp_mean = tf.reduce_mean(log_probs)

            # 计算梯度
            grads = tape.gradient(logp_mean, trainable_vars)

            # 计算Fisher信息（梯度的平方）
            for j, grad in enumerate(grads):
                if grad is not None:
                    fisher_accum[j] = tf.square(grad)

            print(f"Fisher矩阵计算完成，包含 {len(fisher_accum)} 个参数块")
            return fisher_accum

        except Exception as e:
            print(f"计算Fisher矩阵时出错: {e}")
            import traceback
            traceback.print_exc()
            # 返回空的Fisher矩阵
            trainable_vars = self.actor.trainable_variables + self.critic.trainable_variables
            return [tf.zeros_like(var, dtype=tf.float32) for var in trainable_vars]

    def compute_env_fisher_matrix(self, dataset: np.ndarray):
        """計算 Fisher 矩陣"""
        fisher = {}
        optimal_params = {}
        for var in self.actor.trainable_variables:
            fisher[var.name] = np.zeros_like(var.numpy())
            optimal_params[var.name] = var.numpy().copy()

        for x in dataset:
            # 标准化输入形状
            x = x.astype(np.float32)

            # 确保输入形状为 (1, n_features)
            if x.ndim == 1:
                x = x.reshape(1, -1)
            elif x.ndim == 3:
                x = x.reshape(x.shape[0], x.shape[2])
            elif x.ndim == 2 and x.shape[0] != 1:
                x = x.reshape(1, -1)

            with tf.GradientTape() as tape:
                probs = self.actor(x)
                log_prob = tf.math.log(probs + 1e-8)
            grads = tape.gradient(log_prob, self.actor.trainable_variables)
            for g, var in zip(grads, self.actor.trainable_variables):
                if g is not None:
                    fisher[var.name] += (g.numpy() ** 2) / len(dataset)

        self.fisher_matrix = fisher
        self.optimal_params = optimal_params

    def check_variables_initialization(self):
        """检查变量初始化状态"""
        print("=== 变量初始化状态检查 ===")

        variables_to_check = [
            ('optimal_params', self.optimal_params),
            ('fisher_matrices', self.fisher_matrices),
            ('task_memory', self.task_memory),
            ('ewc_means', self.ewc_means),
            ('ewc_fisher', self.ewc_fisher)
        ]

        for var_name, var_value in variables_to_check:
            if hasattr(self, var_name):
                if isinstance(var_value, dict):
                    print(f"✓ {var_name}: 字典类型, 包含 {len(var_value)} 个键")
                elif isinstance(var_value, deque):
                    print(f"✓ {var_name}: 队列类型, 包含 {len(var_value)} 个元素")
                elif var_value is None:
                    print(f"✗ {var_name}: None")
                else:
                    print(f"✗ {var_name}: 意外类型 {type(var_value)}")
            else:
                print(f"✗ {var_name}: 未定义")

        print("=== 检查完成 ===\n")

    def save_task_knowledge(self, task_experiences):
        """保存任务知识 - 安全版本"""
        try:
            # 强制检查并重新初始化变量
            self._safety_check_variables()

            print(f"开始保存任务 {self.current_task_id} 的知识...")

            # 保存最优参数
            optimal_params = []
            for var in self.actor.trainable_variables:
                optimal_params.append(tf.identity(var))
            for var in self.critic.trainable_variables:
                optimal_params.append(tf.identity(var))

            self.optimal_params[self.current_task_id] = optimal_params

            # 计算并保存Fisher信息矩阵
            print("计算Fisher信息矩阵...")
            fisher_matrix = self._compute_fisher_matrix_safe(task_experiences)
            self.fisher_matrices[self.current_task_id] = fisher_matrix

            # 保存任务经验
            print("处理经验数据...")
            if hasattr(self, '_process_experiences_for_memory'):
                processed_experiences = self._process_experiences_for_memory(task_experiences)
                self.task_memory.extend(processed_experiences)

            print(f"任务 {self.current_task_id} 知识已保存")
            print(f"当前保存的任务数量: {len(self.optimal_params)}")

            # 移动到下一个任务
            self.current_task_id += 1

        except Exception as e:
            print(f"保存任务知识时出错: {e}")
            import traceback
            traceback.print_exc()

    def _safety_check_variables(self):
        """安全检查变量"""
        if not hasattr(self, 'optimal_params') or not isinstance(self.optimal_params, dict):
            print("警告: optimal_params 异常，重新初始化")
            self.optimal_params = {}

        if not hasattr(self, 'fisher_matrices') or not isinstance(self.fisher_matrices, dict):
            print("警告: fisher_matrices 异常，重新初始化")
            self.fisher_matrices = {}

        if not hasattr(self, 'task_memory') or self.task_memory is None:
            print("警告: task_memory 异常，重新初始化")
            self.task_memory = deque(maxlen=1000)

    def _compute_fisher_matrix_safe(self, experiences):
        """安全的Fisher矩阵计算"""
        try:
            return self._compute_fisher_matrix(experiences)
        except Exception as e:
            print(f"Fisher矩阵计算失败，使用空矩阵: {e}")
            trainable_vars = self.actor.trainable_variables + self.critic.trainable_variables
            return [tf.zeros_like(var) for var in trainable_vars]

    def _process_experiences_for_memory(self, experiences):
        """处理经验以便存储到长期记忆 - 修复版本"""
        try:
            states, actions, advantages, old_probs, returns = experiences
            processed_experiences = []

            # 确保是numpy数组以便处理
            if hasattr(states, 'numpy'):
                states = states.numpy()
            if hasattr(actions, 'numpy'):
                actions = actions.numpy()
            if hasattr(advantages, 'numpy'):
                advantages = advantages.numpy()
            if hasattr(old_probs, 'numpy'):
                old_probs = old_probs.numpy()
            if hasattr(returns, 'numpy'):
                returns = returns.numpy()

            batch_size = len(states) if hasattr(states, '__len__') else 1

            for i in range(batch_size):
                try:
                    state = states[i] if batch_size > 1 else states
                    action = actions[i] if batch_size > 1 else actions
                    advantage = advantages[i] if batch_size > 1 else advantages
                    old_prob = old_probs[i] if batch_size > 1 else old_probs
                    return_ = returns[i] if batch_size > 1 else returns

                    # 确保数据格式正确
                    state = np.array(state, dtype=np.float32).flatten()
                    action = int(action) if self.is_discrete else float(action)
                    advantage = float(advantage)
                    old_prob = np.array(old_prob, dtype=np.float32).flatten()
                    return_ = float(return_)

                    processed_experiences.append((state, action, advantage, old_prob, return_, self.current_task_id))

                except Exception as e:
                    print(f"处理经验 {i} 时出错: {e}")
                    continue

            print(f"成功处理 {len(processed_experiences)} 条经验")
            return processed_experiences

        except Exception as e:
            print(f"处理经验数据时出错: {e}")
            return []
    def replay_previous_tasks(self, batch_size=32):
        """回放先前任务的记忆 - 修复版本"""
        if len(self.task_memory) == 0:
            return

        try:
            # 随机采样一批记忆
            indices = np.random.choice(len(self.task_memory),
                                       size=min(batch_size, len(self.task_memory)),
                                       replace=False)
            batch = [self.task_memory[i] for i in indices]

            # 安全地解包数据
            states_list, actions_list, advantages_list, old_probs_list, returns_list = [], [], [], [], []

            for exp in batch:
                if len(exp) >= 5:
                    # 状态数据
                    state = np.array(exp[0], dtype=np.float32).flatten()
                    states_list.append(state)

                    # 动作数据
                    action = exp[1]
                    actions_list.append(action)

                    # 优势值
                    advantages_list.append(float(exp[2]))

                    # 旧概率 - 确保是概率分布
                    old_prob = np.array(exp[3], dtype=np.float32).flatten()
                    if len(old_prob) != self.action_dim:
                        # 如果形状不对，调整为均匀分布
                        old_prob = np.ones(self.action_dim, dtype=np.float32) / self.action_dim
                    old_probs_list.append(old_prob)

                    # 回报值
                    returns_list.append(float(exp[4]))

            if len(states_list) == 0:
                return

            # 转换为张量
            states = tf.convert_to_tensor(states_list, dtype=tf.float32)
            actions = tf.convert_to_tensor(actions_list, dtype=tf.int32 if self.is_discrete else tf.float32)
            advantages = tf.convert_to_tensor(advantages_list, dtype=tf.float32)
            old_probs = tf.convert_to_tensor(old_probs_list, dtype=tf.float32)
            returns = tf.convert_to_tensor(returns_list, dtype=tf.float32)

            # 使用修复后的train_step进行训练
            self.train_step(states, actions, advantages, old_probs, returns)

        except Exception as e:
            print(f"回放过程中出现错误: {e}")
            import traceback
            traceback.print_exc()
    def _gaussian_prob(self, mean, std, actions):
        """計算高斯分佈的概率密度（連續動作空間）"""
        # 確保形狀正確
        if len(mean.shape) == 1:
            mean = tf.reshape(mean, [-1, 1])
        if len(std.shape) == 1:
            std = tf.reshape(std, [-1, 1])
        if len(actions.shape) == 1:
            actions = tf.reshape(actions, [-1, 1])

        # 計算高斯概率密度
        var = tf.square(std) + 1e-8
        exponent = -0.5 * tf.square(actions - mean) / var
        norm_factor = 1.0 / tf.sqrt(2.0 * np.pi * var)
        prob = norm_factor * tf.exp(exponent)

        return tf.reduce_prod(prob, axis=1, keepdims=True)

    def reset_ewc_variables(self):
        """重置EWC变量（在开始新训练前调用）"""
        self._initialize_ewc_variables()
        self.current_task_id = 0
        print("EWC变量已重置")


    def test_task_performance(self, env, task_id=None):
        """测试在特定任务上的性能 - 修复版本"""
        try:
            state = env.reset()
            total_reward = 0
            done = False
            steps = 0
            max_steps = 1000  # 防止无限循环

            while not done and steps < max_steps:
                try:
                    # 安全地解包返回值
                    action_result = self.select_action(state)

                    # 检查返回值数量
                    if isinstance(action_result, tuple) and len(action_result) >= 2:
                        action, action_prob = action_result[0], action_result[1]
                    else:
                        # 如果返回值不符合预期，使用默认值
                        print(f"警告: select_action 返回了意外格式: {action_result}")
                        if self.is_discrete:
                            action = 0
                            action_prob = 0.25
                        else:
                            action = np.zeros(self.action_dim)
                            action_prob = 0.5

                    next_state, reward, done, info = env.step(action)
                    total_reward += reward
                    state = next_state
                    steps += 1

                except Exception as e:
                    print(f"测试过程中出错: {e}")
                    break

            print(f"任务测试完成: 总奖励 = {total_reward}, 步数 = {steps}")
            return total_reward

        except Exception as e:
            print(f"测试任务性能时出错: {e}")
            return 0.0
    def _compute_ewc_loss(self):
        ewc_loss = 0.0
        if not self.fisher_matrices:
            return tf.constant(0.0, dtype=tf.float32)

        current_params = self.actor.trainable_variables + self.critic.trainable_variables

        for task_id in self.fisher_matrices:
            fisher_list = self.fisher_matrices[task_id]  # list of tensors (same order)
            optimal_params = self.optimal_params[task_id]  # list of numpy arrays or tensors

            for var, opt_param, fisher in zip(current_params, optimal_params, fisher_list):
                # 确保 opt_param/fisher 为 tf.Tensor
                opt_t = tf.convert_to_tensor(opt_param, dtype=tf.float32)
                fisher_t = tf.convert_to_tensor(fisher, dtype=tf.float32)
                ewc_loss += 0.5 * self.ewc_lambda * tf.reduce_sum(fisher_t * tf.square(var - opt_t))

        return ewc_loss
    def ewc_loss(self):
        # 保存策略网络的参数
        # 初始化 optimal_params，确保它是 tf.Tensor 类型
        if self.optimal_params is None:
            self.optimal_params = [tf.Variable(param) for param in self.policy_params]

        # 将 optimal_params 转换为 tf.Variable 类型
        self.old_policy_params = [tf.Variable(param) for param in self.optimal_params]
        ewc_loss = 0
        for param, old_param, fisher_matrix in zip(self.optimal_params, self.old_policy_params, self.fisher_matrices):
            ewc_loss += tf.reduce_sum(fisher_matrix * (param - old_param) ** 2)
        return self.ewc_lambda * ewc_loss

    def get_policy(self, states):
        """获取策略输出，确保形状正确"""
        policy_output = self.policy(states)

        # 如果是离散动作空间，确保返回 [batch_size, action_dim]
        if self.is_discrete:
            if isinstance(policy_output, (list, tuple)):
                # 如果返回多个值，取第一个作为logits
                logits = policy_output[0]
            else:
                logits = policy_output

            # 确保是二维的
            if len(logits.shape) == 1:
                logits = tf.expand_dims(logits, 0)

            return logits
        else:
            # 连续动作空间处理
            return policy_output

    def train_step_onehot(
            self, states, actions, advantages, old_probs, returns,
            clip_ratio=0.1, use_ewc=False
    ):
        """执行一次 PPO 更新 (自动处理 old_probs shape)"""

        # === 转换为 Tensor ===
        states = tf.convert_to_tensor(states, dtype=tf.float32)  # (N, state_dim)
        actions = tf.convert_to_tensor(actions, dtype=tf.int32)  # (N,)
        advantages = tf.convert_to_tensor(advantages, dtype=tf.float32)  # (N,)
        returns = tf.convert_to_tensor(returns, dtype=tf.float32)  # (N,)

        old_probs = tf.convert_to_tensor(old_probs, dtype=tf.float32)

        # === one-hot 动作 ===
        action_onehot = tf.one_hot(actions, depth=self.action_dim, dtype=tf.float32)  # (N, action_dim)

        # === 修复 old_probs ===
        if len(old_probs.shape) == 1:
            # (N,) -> 说明存的是已选动作的概率
            old_probs_selected = old_probs
        elif len(old_probs.shape) == 2 and old_probs.shape[1] == self.action_dim:
            # (N, action_dim) -> 存的是完整分布
            old_probs_selected = tf.reduce_sum(old_probs * action_onehot, axis=1)
        else:
            raise ValueError(f"old_probs shape 不合法: {old_probs.shape}")

        with tf.GradientTape(persistent=True) as tape:
            # 策略网络前向
            logits = self.actor(states, training=True)  # (N, action_dim)
            probs = tf.nn.softmax(logits)  # (N, action_dim)

            # 新策略下的已选动作概率
            new_probs = tf.reduce_sum(probs * action_onehot, axis=1)  # (N,)

            # 比例 r_t
            ratio = new_probs / (old_probs_selected + 1e-8)

            # PPO clip surrogate
            surrogate1 = ratio * advantages
            surrogate2 = tf.clip_by_value(ratio, 1 - clip_ratio, 1 + clip_ratio) * advantages
            actor_loss = -tf.reduce_mean(tf.minimum(surrogate1, surrogate2))

            # 值函数损失
            values = self.critic(states, training=True)  # (N,1)
            value_loss = tf.reduce_mean((returns - tf.squeeze(values)) ** 2)

            # 熵正则
            entropy = -tf.reduce_mean(tf.reduce_sum(probs * tf.math.log(probs + 1e-8), axis=1))

            # EWC 正则化
            ewc_loss = self._compute_ewc_loss() if use_ewc and self.fisher_matrices else 0.0

            total_loss = actor_loss + 0.5 * value_loss - 0.01 * entropy + ewc_loss

        # === 更新梯度 ===
        actor_grads = tape.gradient(total_loss, self.actor.trainable_variables)
        critic_grads = tape.gradient(total_loss, self.critic.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))
        self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_variables))
        del tape

        return total_loss, actor_loss, value_loss, entropy,ewc_loss

    def train_step(self, states, actions, advantages, old_log_probs, returns,
                   clip_ratio=0.1, use_ewc=False,
                   adv_clip=10.0, logprob_clip=20.0, ratio_clip_max=10.0,
                   value_clip=10.0, entropy_clip=5.0, grad_clip_norm=5.0,
                   normalize_returns=False):
        """受保护的 PPO 更新，平衡 policy_loss 和 value_loss"""

        # 转张量
        states = tf.convert_to_tensor(states, dtype=tf.float32)
        advantages = tf.convert_to_tensor(advantages, dtype=tf.float32)
        old_log_probs = tf.convert_to_tensor(old_log_probs, dtype=tf.float32)
        returns = tf.convert_to_tensor(returns, dtype=tf.float32)

        if normalize_returns:
            returns = (returns - tf.reduce_mean(returns)) / (tf.math.reduce_std(returns) + 1e-8)

        if self.is_discrete:
            actions = tf.convert_to_tensor(actions, dtype=tf.int32)
            if len(actions.shape) > 1:
                actions = tf.squeeze(actions, axis=-1)
        else:
            actions = tf.convert_to_tensor(actions, dtype=tf.float32)

        with tf.GradientTape() as tape:
            # --- policy & log_probs ---
            if self.is_discrete:
                logits = self.get_policy(states)
                dist = tfp.distributions.Categorical(logits=logits)
                new_log_probs = dist.log_prob(actions)
                actions_one_hot = tf.one_hot(actions, depth=self.action_dim)
                #old_action_probs = tf.reduce_sum(old_probs * actions_one_hot, axis=1)
                #old_log_probs = tf.math.log(old_action_probs + 1e-8)
                new_log_probs = tf.reshape(new_log_probs, [-1, 1])
                #old_log_probs = tf.reshape(old_log_probs, [-1, 1])
            else:
                policy_output = self.get_policy(states)
                if isinstance(policy_output, (list, tuple)) and len(policy_output) == 2:
                    mean, log_std = policy_output
                else:
                    mean, log_std = policy_output, self.log_std
                std = tf.exp(log_std)
                dist = tfp.distributions.Normal(mean, std)
                new_log_probs = tf.reduce_sum(dist.log_prob(actions), axis=-1, keepdims=True)
                #old_log_probs = tf.reshape(old_log_probs, [-1, 1])

            # --- clip log_probs & ratio ---
            new_log_probs = tf.clip_by_value(new_log_probs, -logprob_clip, logprob_clip)
            old_log_probs = tf.clip_by_value(old_log_probs, -logprob_clip, logprob_clip)

            ratio = tf.exp(new_log_probs - old_log_probs)
            ratio = tf.clip_by_value(ratio, 0.0, ratio_clip_max)

            # --- clip advantages ---
            advantages = tf.reshape(advantages, [-1, 1])
            advantages = tf.clip_by_value(advantages, -adv_clip, adv_clip)

            # --- surrogate loss ---
            surrogate1 = ratio * advantages
            surrogate2 = tf.clip_by_value(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * advantages
            policy_loss = -tf.reduce_mean(tf.minimum(surrogate1, surrogate2))

            # --- value loss (裁剪 & 可归一化) ---
            values = self.value_net(states)
            returns = tf.reshape(returns, [-1, 1])
            value_diff = tf.clip_by_value(returns - values, -value_clip, value_clip)
            value_loss = tf.reduce_mean(tf.square(value_diff))

            # --- entropy loss ---
            entropy = tf.reduce_mean(tf.clip_by_value(dist.entropy(), 0.0, entropy_clip))

            # --- 总 loss ---
            loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

            # --- EWC loss ---
            if use_ewc:
                ewc_loss_val = self._compute_ewc_loss()
                loss += ewc_loss_val
            else:
                ewc_loss_val = tf.constant(0.0)

        # --- 梯度裁剪 ---
        trainable_vars = self.policy_params + self.value_params
        grads = tape.gradient(loss, trainable_vars)
        grads = [tf.clip_by_norm(g, grad_clip_norm) if g is not None else None for g in grads]
        grads_and_vars = [(g, v) for g, v in zip(grads, trainable_vars) if g is not None]
        if grads_and_vars:
            self.optimizer.apply_gradients(grads_and_vars)

        # --- 返回各项 loss ---
        return loss.numpy(), policy_loss.numpy(), value_loss.numpy(), entropy.numpy(), ewc_loss_val.numpy()

    def train_on_batch(self, states, actions, old_probs, returns, values,
                       clip_ratio=0.1, use_ewc=False, normalize_adv=True):
        """高层接口：接收 rollout 数据并调用 train_step"""
        try:
            # 保证输入维度正确
            states = self._ensure_2d(states, self.state_dim)

            # 计算优势函数
            advantages = self._compute_advantages(returns, values)

            # 归一化优势（推荐）
            if normalize_adv:
                advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)

            # 调用底层 train_step
            loss, policy_loss, value_loss, entropy, ewc_loss = self.train_step(
                states, actions, advantages, old_probs, returns,
                clip_ratio=clip_ratio, use_ewc=use_ewc
            )

            return loss, policy_loss, value_loss, entropy, ewc_loss

        except Exception as e:
            print(f"train_on_batch 出错: {e}")
            import traceback
            traceback.print_exc()
            return None, None, None, None, None

    def _ensure_2d(self, data, expected_dim):
        """确保数据是2D形状 (batch_size, feature_dim)"""
        if not isinstance(data, np.ndarray):
            data = np.array(data)

        if len(data.shape) == 1:
            if len(data) == expected_dim:
                # 单个样本
                return data.reshape(1, -1)
            else:
                # 批量样本但只有一维
                return data.reshape(-1, expected_dim)
        return data

    def _compute_advantages(self, returns, values):
        """计算优势函数"""
        advantages = returns - values
        return advantages
    def collect_experience(self, env, num_episodes):
        for episode in range(num_episodes):
            state = env.reset()
            done = False

            while not done:
                # 获取动作概率
                action_probs = self.actor.predict(state.reshape(1, -1), verbose=0)[0]

                # 选择动作（确保是标量）
                action = np.random.choice(len(action_probs), p=action_probs)

                # 执行动作
                next_state, reward, done, _ = env.step(action)

                # 存储经验（确保动作是标量）
                self.store_experience(state, action, action_probs, reward, done, self.pointer)

                state = next_state
                self.pointer = (self.pointer + 1) % self.buffer_size

    def store_experience(self, state, action, advantage, old_prob, return_, next_state=None):
        """存储经验数据 - 确保概率格式正确"""
        try:
            # 标准化状态形状
            state = np.array(state, dtype=np.float32).flatten()
            if len(state) != self.state_dim:
                state = state[:self.state_dim] if len(state) > self.state_dim else np.pad(state, (0,
                                                                                                  self.state_dim - len(
                                                                                                      state)))

            # 处理动作
            if self.is_discrete:
                action = int(action)
            else:
                action = np.array(action, dtype=np.float32).flatten()
                if len(action) != self.action_dim:
                    action = action[:self.action_dim] if len(action) > self.action_dim else np.pad(action, (0,
                                                                                                            self.action_dim - len(
                                                                                                                action)))

            # 关键修复：确保 old_prob 是概率分布格式
            old_prob = np.array(old_prob, dtype=np.float32)

            if self.is_discrete:
                # 对于离散动作，old_prob 应该是所有动作的概率分布 [action_dim]
                if old_prob.ndim == 0:
                    # 如果是标量，转换为均匀分布
                    old_prob = np.ones(self.action_dim) / self.action_dim
                elif old_prob.ndim == 1 and len(old_prob) == 1:
                    # 如果是一维但只有一个元素，也转换为均匀分布
                    old_prob = np.ones(self.action_dim) / self.action_dim
                elif old_prob.ndim == 1 and len(old_prob) != self.action_dim:
                    # 如果长度不匹配，调整大小
                    if len(old_prob) > self.action_dim:
                        old_prob = old_prob[:self.action_dim]
                    else:
                        old_prob = np.pad(old_prob, (0, self.action_dim - len(old_prob)))
                        # 归一化
                        old_prob = old_prob / np.sum(old_prob) if np.sum(old_prob) > 0 else np.ones(
                            self.action_dim) / self.action_dim
            else:
                # 连续动作空间，保持原样
                pass

            # 存储经验
            experience = (state, action, float(advantage), old_prob, float(return_))
            self.task_memory.append(experience)

            print(f"存储经验: 动作{action}, old_prob形状{old_prob.shape}, 和{np.sum(old_prob):.3f}")

        except Exception as e:
            print(f"存储经验时出错: {e}")
            import traceback
            traceback.print_exc()

    def debug_select_action(self, state):
        """调试版本的select_action，显示详细信息"""
        print(f"=== select_action 调试 ===")
        print(f"输入状态形状: {np.array(state).shape}")
        print(f"是否离散动作: {self.is_discrete}")

        state_tensor = tf.expand_dims(tf.convert_to_tensor(state, dtype=tf.float32), 0)
        print(f"状态张量形状: {state_tensor.shape}")

        if self.is_discrete:
            logits = self.actor(state_tensor)
            print(f"Logits形状: {logits.shape}")

            dist = tfp.distributions.Categorical(logits=logits)
            action = dist.sample()[0].numpy()
            action_prob = dist.prob(action).numpy()

            print(f"动作: {action}, 概率: {action_prob}")
            print(f"动作类型: {type(action)}, 概率类型: {type(action_prob)}")

            result = (int(action), float(action_prob))
        else:
            mean = self.actor(state_tensor)
            print(f"均值形状: {mean.shape}")

            log_std = tf.ones_like(mean) * self.log_std
            dist = tfp.distributions.Normal(loc=mean, scale=tf.exp(log_std))
            action = dist.sample()[0].numpy()
            action_prob = dist.prob(action).numpy()
            action_prob = np.prod(action_prob)

            print(f"动作: {action}, 概率: {action_prob}")
            result = (action, float(action_prob))

        print(f"返回值: {result}")
        print(f"返回值长度: {len(result)}")
        print("=== 调试结束 ===\n")

        return result

    # 临时替换select_action进行调试
    def test_task_performance_debug(self, env, task_id=None):
        """调试版本的任务性能测试"""
        state = env.reset()
        total_reward = 0
        done = False

        print("开始调试任务性能测试...")

        while not done:
            # 使用调试版本
            action, prob = self.debug_select_action(state)
            print(f"选择的动作: {action}, 概率: {prob}")

            next_state, reward, done, _ = env.step(action)
            total_reward += reward
            state = next_state

        return total_reward


    def select_action(self,   state):

        # --- 确保 state 为张量 ---
        if state is None or (isinstance(state, str) and state == ''):
            state = np.zeros(self.state_dim, dtype=np.float32)
        state = np.array(state, dtype=np.float32)
        if len(state.shape) == 1:
            state = state[None, :]  # batch 维度

        state_tensor = tf.convert_to_tensor(state, dtype=tf.float32)

        # --- 计算 policy & value ---
        value = self.value_net(state_tensor)
        value = tf.squeeze(value).numpy()  # 去掉多余维度

        if self.is_discrete:
            logits = self.get_policy(state_tensor)
            dist = tfp.distributions.Categorical(logits=logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            action = tf.squeeze(action).numpy()
            log_prob = tf.squeeze(log_prob).numpy()

            # 防止类型问题
            if np.isscalar(action):
                action = int(action)
            else:
                action = action.astype(int)
        else:
            mean, log_std = self.get_policy(state_tensor)
            std = tf.exp(log_std)
            dist = tfp.distributions.Normal(mean, std)
            action = dist.sample()
            log_prob = tf.reduce_sum(dist.log_prob(action), axis=-1)

            # mean, log_std = agent.get_policy(state_tensor)
            # std = tf.exp(log_std)
            # dist = tfp.distributions.Normal(mean, std)
            # action = dist.sample()
            # action  = tf.nn.softmax(action).numpy().flatten()
            # action_prob = tf.reduce_prod(dist.prob(action), axis=-1)
            # action_prob = tf.squeeze(action_prob).numpy()

            action = tf.squeeze(action).numpy()
            # 防止空值
            if np.any(np.isnan(action)):
                action = np.zeros(self.action_dim, dtype=np.float32)
            if np.any(np.isnan(log_prob)):
                log_prob = 1.0

        return action, log_prob, value


class ESP32PPOAgent(PPOBaseAgent):
    """專為ESP32設計的輕量級PPO代理,繼承自LifelongPPOAgent"""

    def __init__(self, fisher_matrix=None, optimal_params=None, state_dim=5, action_dim=4,
                 clip_epsilon=0.2, value_coef=0.5, entropy_coef=0.01,
                 ewc_lambda=500, memory_size=1000, hidden_units=32,
                 clip_ratio=0.2, actor_lr=1e-4, critic_lr=1e-3, is_discrete=False,
                 gamma=0.99, lam=0.95):
        super().__init__( fisher_matrix , optimal_params , state_dim , action_dim ,
                 clip_epsilon , value_coef , entropy_coef ,
                 ewc_lambda , memory_size , hidden_units ,
                 clip_ratio , actor_lr, critic_lr , is_discrete ,
                 gamma , lam   )


        self._tflite_models = {}
        # 重新構建更小的網絡


        self.is_discrete=is_discrete
        print(f"ESP32代理初始化完成: {hidden_units}隱藏單元")


    def _build_models(self):
        """显式构建模型"""
        # 创建虚拟输入来构建模型
        dummy_input = tf.zeros((1, self.state_dim))
        
        # 前向传播一次以构建模型
        _ = self.policy(dummy_input)
        _ = self.critic(dummy_input)
        
        print("模型构建完成")
    def count_agent_params(self):
        """安全统计代理参数数量"""
        try:
            actor_params = self.policy.count_params()
            critic_params = self.critic.count_params()
            total_params = actor_params + critic_params
            
            print(f"Actor网络参数: {actor_params}")
            print(f"Critic网络参数: {critic_params}")
            print(f"总参数数量: {total_params}")
            
            return actor_params, critic_params, total_params
            
        except ValueError as e:
            if "isn't built" in str(e):
                print("模型未构建，正在构建...")
                self._build_models()
                return self.count_agent_params()
            else:
                raise e

    def save_tflite_model(self, filepath, model_type='actor'):
        """保存TFLite模型到文件"""
        if model_type not in self._tflite_models:
            self.convert_to_tflite(model_type)

        # 確保目錄存在
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)

        with open(filepath, 'wb') as f:
            f.write(self._tflite_models[model_type])
        print(f"TFLite {model_type} model saved to {filepath}")

    def load_tflite_model(self, filepath, model_type='actor'):
        """從文件加載TFLite模型"""
        with open(filepath, 'rb') as f:
            tflite_model = f.read()

        self._tflite_models[model_type] = tflite_model
        return tflite_model

    def convert_to_tflite(self, model_type='actor', quantize=False, optimize_size=False):
        if model_type == 'actor':
            model = self.actor
        elif model_type == 'critic':
            model = self.critic
        else:
            raise ValueError("model_type must be 'actor' or 'critic'")

        converter = tf.lite.TFLiteConverter.from_keras_model(model)

        if optimize_size:
            converter.optimizations = [tf.lite.Optimize.DEFAULT]

            if quantize:
                # 用 float16 权重量化，ESP32 float32 内核也能跑
                converter.target_spec.supported_types = [tf.float16]

            # 删除 SELECT_TF_OPS
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]

        try:
            tflite_model = converter.convert()
            self._tflite_models[model_type] = tflite_model

            model_size_kb = len(tflite_model) / 1024
            print(f"{model_type} TFLite模型轉換完成: {model_size_kb:.2f} KB")

            return tflite_model

        except Exception as e:
            print(f"TFLite轉換失敗: {e}")
            return None

    def get_model_size_info(self):
        """獲取模型大小信息"""
        size_info = {}

        for model_type in ['actor', 'critic']:
            if model_type in self._tflite_models:
                model_size = len(self._tflite_models[model_type])
                size_info[model_type] = {
                    'bytes': model_size,
                    'kb': model_size / 1024,
                    'params': self._count_params(getattr(self, model_type))
                }

        return size_info

    
    def predict_with_tflite(self, state, model_type='actor'):
        """使用TFLite模型進行預測"""
        if model_type not in self._tflite_models:
            self.convert_to_tflite(model_type)

        try:
            interpreter = tf.lite.Interpreter(
                model_content=self._tflite_models[model_type],
                num_threads=1
            )
            interpreter.allocate_tensors()

            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()

            input_shape = input_details[0]['shape']
            input_data = np.array(state, dtype=np.float32).reshape(input_shape)

            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()

            output_data = interpreter.get_tensor(output_details[0]['index'])
            return output_data[0]

        except Exception as e:
            print(f"TFLite預測錯誤: {e}")
            if model_type == 'actor':
                return self.actor.predict(np.array([state]))[0]
            else:
                return self.critic.predict(np.array([state]))[0]

    def get_esp32_action(self, state,return_probs=False):
        """專為ESP32設計的動作獲取方法"""
        try:
            probs = self.predict_with_tflite(state, 'actor')
            action = (probs > 0.5).astype(np.float32)
            return action
        except:
            return super().get_action(state)

    def get_esp32_value(self, state):
        """專為ESP32設計的值預測方法"""
        try:
            return self.predict_with_tflite(state, 'critic')[0]
        except:
            return self.critic.predict(np.array([state]))[0][0]

    def export_for_esp32(self, base_path="ppo_model",task_id=0):
        """導出ESP32所需的所有文件"""
        os.makedirs(base_path, exist_ok=True)

        # 轉換並保存TFLite模型
        self.actor=self.convert_to_tflite(model_type='actor', quantize=False, optimize_size=False)
        self.critic=self.convert_to_tflite(model_type='critic', quantize=False, optimize_size=False)

        # 使用正確的方法名
        self.save_tflite_model(f"{base_path}/actor_task{task_id}.tflite", 'actor')
        self.save_tflite_model(f"{base_path}/critic_task{task_id}.tflite", 'critic')

        # 生成C頭文件
        self._generate_c_header(f"{base_path}/actor_task{task_id}.tflite", f"{base_path}/actor_task{task_id}.h", 'actor_model')
        self._generate_c_header(f"{base_path}/critic_task{task_id}.tflite", f"{base_path}/critic_task{task_id}.h", 'critic_model')

        # 生成示例代碼
        self._generate_example_code(base_path)

        print(f"ESP32模型已導出到: {base_path}")

    def _generate_c_header(self, tflite_path, header_path, var_name):
        """生成C頭文件"""
        try:
            with open(tflite_path, 'rb') as f:
                model_data = f.read()

            with open(header_path, 'w') as f:
                f.write(f"#ifndef {var_name.upper()}_H\n")
                f.write(f"#define {var_name.upper()}_H\n\n")
                f.write(f"#include <stdint.h>\n\n")
                f.write(f"extern const uint8_t {var_name}[];\n")
                f.write(f"extern const unsigned int {var_name}_len;\n\n")
                f.write(f"#endif\n")

            # 創建實現文件
            impl_path = header_path.replace('.h', '.c')
            with open(impl_path, 'w') as f:
                f.write(f'#include "{os.path.basename(header_path)}"\n\n')
                f.write(f"const uint8_t {var_name}[] = {{\n")

                for i, byte in enumerate(model_data):
                    if i % 12 == 0:
                        f.write("\n  ")
                    f.write(f"0x{byte:02x}, ")

                f.write("\n};\n\n")
                f.write(f"const unsigned int {var_name}_len = {len(model_data)};\n")

            print(f"C頭文件生成: {header_path}")

        except Exception as e:
            print(f"生成C頭文件失敗: {e}")

    def _generate_example_code(self, base_path):
        """生成ESP32示例代碼"""
        example_code = '''// ESP32 Plant HVAC Control Example
#include <TensorFlowLite.h>
#include "actor.h"
#include "critic.h"

// 錯誤報告器
tflite::MicroErrorReporter error_reporter;

// 加載模型
const tflite::Model* actor_model = tflite::GetModel(actor_model);
const tflite::Model* critic_model = tflite::GetModel(critic_model);

// 創建解釋器
static tflite::AllOpsResolver resolver;
static tflite::MicroInterpreter actor_interpreter(actor_model, resolver, tensor_arena, tensor_arena_size, &error_reporter);
static tflite::MicroInterpreter critic_interpreter(critic_model, resolver, tensor_arena, tensor_arena_size, &error_reporter);

// 張量緩衝區（根據模型大小調整）
const int tensor_arena_size = 20 * 1024;
uint8_t tensor_arena[tensor_arena_size];

void setup() {
  Serial.begin(115200);

  // 分配內存
  if (actor_interpreter.AllocateTensors() != kTfLiteOk) {
    Serial.println("Actor模型內存分配失敗");
    return;
  }

  if (critic_interpreter.AllocateTensors() != kTfLiteOk) {
    Serial.println("Critic模型內存分配失敗");
    return;
  }

  Serial.println("ESP32 Plant HVAC Controller Ready");
}

void get_action(float state[5], float action[4]) {
  // 獲取輸入張量
  TfLiteTensor* input = actor_interpreter.input(0);

  // 複製狀態數據
  for (int i = 0; i < 5; i++) {
    input->data.f[i] = state[i];
  }

  // 運行推理
  if (actor_interpreter.Invoke() != kTfLiteOk) {
    Serial.println("Actor推理失敗");
    return;
  }

  // 獲取輸出並二值化
  TfLiteTensor* output = actor_interpreter.output(0);
  for (int i = 0; i < 4; i++) {
    action[i] = (output->data.f[i] > 0.5) ? 1.0 : 0.0;
  }
}

float get_value(float state[5]) {
  // 獲取輸入張量
  TfLiteTensor* input = critic_interpreter.input(0);

  // 複製狀態數據
  for (int i = 0; i < 5; i++) {
    input->data.f[i] = state[i];
  }

  // 運行推理
  if (critic_interpreter.Invoke() != kTfLiteOk) {
    Serial.println("Critic推理失敗");
    return 0.0;
  }

  // 獲取輸出
  TfLiteTensor* output = critic_interpreter.output(0);
  return output->data.f[0];
}

void loop() {
  // 示例狀態數據
  float state[5] = {0.0, 25.0, 0.5, 500.0, 600.0};
  float action[4];

  // 獲取動作
  get_action(state, action);

  // 獲取值預測
  float value = get_value(state);

  // 輸出結果
  Serial.print("Action: ");
  for (int i = 0; i < 4; i++) {
    Serial.print(action[i]);
    Serial.print(" ");
  }
  Serial.print("Value: ");
  Serial.println(value);

  delay(1000);
}
'''

        with open(f"{base_path}/esp32_example.ino", 'w') as f:
            f.write(example_code)

        print(f"示例代碼生成: {base_path}/esp32_example.ino")

    def compress_for_esp32(self, target_size_kb=50):
        """壓縮模型以適應ESP32內存限制"""
        print("開始壓縮模型...")

        current_units = self.hidden_units

        while current_units >= 4:
            self.hidden_units = current_units
            self.actor = self._build_actor()
            self.critic = self._build_critic()

            self.convert_to_tflite('actor')
            self.convert_to_tflite('critic')

            size_info = self.get_model_size_info()
            total_size_kb = sum([info['kb'] for info in size_info.values()])

            print(f"隱藏單元: {current_units}, 總大小: {total_size_kb:.2f} KB")

            if total_size_kb <= target_size_kb:
                print(f"模型壓縮完成！最終大小: {total_size_kb:.2f} KB")
                return True

            current_units -= 2

        print("無法壓縮到目標大小")
        return False
 

    def save_fisher_and_params(self, path: str):
        np.savez_compressed(
            path,
            fisher={k: v for k,v in self.fisher_matrix.items()},
            optimal={k: v for k,v in self.optimal_params.items()}
        )
        print(f"  Fisher matrix & optimal params saved to {path}")


    @staticmethod
    def process_experiences(agent, experiences, gamma=0.99, gae_lambda=0.95):
        """
        处理经验并计算 advantages 和 returns
        """
        # 提取数据
        states = [exp["state"] for exp in experiences]
        actions = [exp["action"] for exp in experiences]
        rewards = [exp["reward"] for exp in experiences]
        next_states = [exp["next_state"] for exp in experiences]
        dones = [exp["done"] for exp in experiences]
        old_probs = [exp["old_prob"] for exp in experiences]

        # 转换为 Tensor
        states = tf.stack(states)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
        dones = tf.convert_to_tensor(dones, dtype=tf.float32)
        old_probs = tf.stack(old_probs)

        # === 关键修正: 确保 actions 是索引 (batch,) ===
        actions = tf.stack(actions)
        if len(actions.shape) == 2:  # (batch, action_dim) -> one-hot
            actions = tf.argmax(actions, axis=1, output_type=tf.int32)
        else:  # (batch,) -> 已经是索引
            actions = tf.cast(actions, tf.int32)

        # 使用 critic 网络计算状态价值
        values = tf.squeeze(agent.critic(states, training=False), axis=1)  # (batch,)
        next_values = tf.squeeze(agent.critic(tf.stack(next_states), training=False), axis=1)  # (batch,)

        # 计算 advantages
        advantages = agent.compute_advantages(rewards, values, next_values, dones, gamma, gae_lambda)

        # 计算 returns
        returns = agent.compute_returns(rewards, dones, gamma)

        return states, actions, advantages, old_probs, returns

    @staticmethod
    def compute_returns(rewards, dones, gamma=0.99):
        """计算折扣回报"""
        returns = []
        R = 0
        for reward, done in zip(reversed(rewards), reversed(dones)):
            if done:
                R = reward
            else:
                R = reward + gamma * R
            returns.insert(0, R)
        return tf.convert_to_tensor(returns, dtype=tf.float32)

    @staticmethod
    def compute_advantages(rewards, values, next_values, dones, gamma=0.99, gae_lambda=0.95):
        """
        计算GAE优势函数

        参数:
        rewards: 奖励序列 [r0, r1, r2, ..., r_{T-1}]
        values: 状态价值序列 [V(s0), V(s1), V(s2), ..., V(s_{T-1})]
        next_values: 下一个状态价值序列 [V(s1), V(s2), V(s3), ..., V(sT)]
        dones: 是否结束序列 [done0, done1, ..., done_{T-1}]
        gamma: 折扣因子
        gae_lambda: GAE参数
        """
        advantages = []
        advantage = 0

        # 从后向前计算
        for t in reversed(range(len(rewards))):
            if dones[t]:
                # 如果回合结束，下一个状态价值为0
                next_value = 0
            else:
                next_value = next_values[t]

            # TD误差：r + γ*V(s') - V(s)
            delta = rewards[t] + gamma * next_value - values[t]

            # GAE: advantage = δ + (γλ)*advantage_{t+1}
            advantage = delta + gamma * gae_lambda * advantage
            advantages.insert(0, advantage)  # 插入到开头

        return tf.convert_to_tensor(advantages, dtype=tf.float32)

    @staticmethod
    def collect_experiences(agent, env, num_episodes=100, max_steps_per_episode=None):
        """
        收集智能体在环境中的经验

        参数:
            agent: 智能体对象，需要有 get_action 方法
            env: 环境对象
            num_episodes: 要收集的回合数
            max_steps_per_episode: 每个回合的最大步数，如果为None则使用env.seq_len

        返回:
            experiences: 收集到的经验列表
        """
        if max_steps_per_episode is None:
            max_steps_per_episode = env.seq_len

        experiences = []

        for episode in range(num_episodes):
            state = env.reset()
            episode_experiences = []
            done = False
            step = 0

            # 随机选择或根据当前模式设置真实标签
            if env.mode == "flowering":
                true_label = 2
            elif env.mode == "seeding":
                true_label = 1
            else:  # growing
                true_label = 0

            print(f"开始收集第 {episode + 1}/{num_episodes} 回合经验 (模式: {env.mode})")

            while not done and step < max_steps_per_episode:
                # 智能体选择动作
                #action = agent.get_action(state)
                action, old_prob = agent.get_action(state )

                # 在环境中执行动作
                next_state, reward, done, info = env.step(
                    action,
                    params={
                        "energy_penalty": 0.1,
                        "switch_penalty_per_toggle": 0.2,
                        "vpd_penalty": 2.0,
                        "flower_bonus": 0.5,
                        "seed_bonus": 0.5,
                        "grow_bonus": 0.5
                    },
                    true_label=true_label
                )

                # 存储经验
                experience = {
                    'state': state.copy(),
                    'action': action,
                    'old_prob': old_prob,
                    'reward': reward,
                    'next_state': next_state.copy(),
                    'done': done,
                    'info': info,
                    'true_label': true_label,
                    'episode': episode,
                    'step': step
                }

                episode_experiences.append(experience)

                # 更新状态
                state = next_state
                step += 1

                # 打印进度
                if step % 10 == 0:
                    health_status = ["健康", "不健康", "無法判定"][info['health_status']]
                    print(f"  步骤 {step}: 状态={health_status}, 奖励={reward:.3f}, "
                          f"温度={info['temp']:.1f}°C, 湿度={info['humid']:.3f}")

            # 将本回合的经验添加到总经验中
            experiences.extend(episode_experiences)
            # 存储经验（确保动作是标量）

            # 打印回合总结
            total_reward = sum(exp['reward'] for exp in episode_experiences)
            avg_health = np.mean([exp['info']['health_status'] for exp in episode_experiences])
            health_percentage = np.mean([1 if exp['info']['health_status'] == 0 else 0
                                         for exp in episode_experiences]) * 100

            print(f"回合 {episode + 1} 完成: 总奖励={total_reward:.3f}, "
                  f"健康时间比例={health_percentage:.1f}%, 步数={step}")
            print("-" * 50)

        return experiences



class ESP32OnlinePPOFisherAgent(ESP32PPOAgent):
    """專為ESP32設計的輕量級PPO代理 支持 online EWC 和 TFLite 導出"""

    def __init__(self, fisher_matrix=None, optimal_params=None, state_dim=5, action_dim=4,
                 clip_epsilon=0.2, value_coef=0.5, entropy_coef=0.01,
                 ewc_lambda=500, memory_size=1000, hidden_units=32,  # 增加到 32
                 clip_ratio=0.2, actor_lr=1e-4, critic_lr=1e-3, is_discrete=False,  # 降低actor学习率
                 gamma=0.99, lam=0.95):
        super().__init__(fisher_matrix , optimal_params , state_dim , action_dim ,
                 clip_epsilon , value_coef , entropy_coef ,
                 ewc_lambda , memory_size , hidden_units ,
                 clip_ratio , actor_lr, critic_lr , is_discrete ,
                 gamma , lam )

        self.hidden_units = hidden_units
        self._tflite_models = {}
        #self.actor = self._build_actor()
        #self.critic = self._build_critic()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
        #self.value_net = self.critic

        self.logits_layer = self.actor.layers[-1]

        self.policy_params = self.actor.trainable_variables
        self.value_params = self.critic.trainable_variables
        # EWC online
        self.online_fisher = fisher_matrix
        self.optimal_params = optimal_params
        self.ema_decay = 0.99
        self.fisher_update_frequency = 1
        self.update_counter = 0
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.clip_ratio = clip_ratio
        self.gamma = gamma
        self.lam = lam
        self.vf_coef = 0.5
        self.ent_coef = 0.01
        self.actor_opt = tf.keras.optimizers.Adam(actor_lr)
        self.critic_opt = tf.keras.optimizers.Adam(critic_lr)
        self.is_discrete = is_discrete
        print(f"ESP32代理初始化完成: {hidden_units}隱藏單元")

    def sigmoid(self, x):
        return 1.0 / (1.0 + tf.exp(-x))

    # ================= 收集一条 trajectory =================
    def collect_trajectory(self, env, max_steps=200):
        states, actions, rewards, log_probs, values ,dones= [], [], [], [], [],[]

        #state = env.reset()
        #prev_action=state.prev_action
        #state_vec= [state.health,state.temp,state.humid,state.light,state.co2,state.ph, state.water]
        health, temp, humid, light, co2, ph, water,*prev_action=env.reset()
        state = [health, temp, humid, light, co2, ph, water]
        for _ in range(max_steps):
            action, log_prob = self.get_action(state )
            next_state, reward, done, _ = env.step(action)

            states.append(state)
            actions.append(action)
            rewards.append(reward)
            log_probs.append(log_prob)
            values.append(self.critic(np.expand_dims(state, 0)).numpy()[0, 0])
            dones.append(done)
            state = next_state
            if done:
                break

        # 最后状态的 V(s_T) 作为 bootstrap
        last_value = self.critic(np.expand_dims(state, 0)).numpy()[0, 0]
        values.append(last_value)

        return np.array(states, dtype=np.float32), \
            np.array(actions, dtype=np.float32), \
            np.array(rewards, dtype=np.float32), \
            np.array(log_probs, dtype=np.float32), \
            np.array(values, dtype=np.float32), \
            np.array(dones, dtype=np.float32)

    # ================= 批量收集多条 trajectory =================
    def collect_trajectories(self, env, num_episodes=10, max_steps=200):
        all_states, all_actions, all_rewards, all_log_probs, all_values,all_dones = [], [], [], [], [], []

        for _ in range(num_episodes):
            states, actions, rewards, log_probs, values,dones = self.collect_trajectory(env, max_steps)
            all_states.append(states)
            all_actions.append(actions)
            all_rewards.append(rewards)
            all_log_probs.append(log_probs)
            all_values.append(values)
            all_dones.append(dones)

        return all_states, all_actions, all_rewards, all_log_probs, all_values,all_dones

    # ================= GAE 优势函数计算 =================
    def compute_gae(self, rewards, values, dones, gamma=0.99, lam=0.95):
        rewards = np.array(rewards, dtype=np.float32)
        dones = np.array(dones, dtype=np.float32)
        values = np.array(values, dtype=np.float32)  # 長度 = len(rewards)+1

        adv = np.zeros_like(rewards, dtype=np.float32)
        gae = 0
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + gamma * (1 - dones[t]) * values[t + 1] - values[t]
            gae = delta + gamma * lam * (1 - dones[t]) * gae
            adv[t] = gae

        returns = adv + values[:-1]  # 去掉最後的 bootstrap
        return adv, returns

    def reset_buffer(self):
        """清空经验缓存"""
        self.states = []
        self.actions = []
        self.rewards = []
        self.action_probs = []
        self.next_states = []
        self.dones = []

    def store_transition(self, state, action, reward, action_prob, next_state=None, done=False):
        """存一条经验"""
        self.states.append(np.array(state, copy=False))
        self.actions.append(np.array(action, copy=False))
        self.rewards.append(reward)
        self.action_probs.append(action_prob)
        self.next_states.append(np.array(next_state if next_state is not None else state, copy=False))
        self.dones.append(done)

    def get_buffer_items(self,ppo_buf):
        """返回 numpy 格式的 batch"""
           
        return (
            np.array(ppo_buf.states, dtype=np.float32),
            np.array(ppo_buf.actions),
            np.array(ppo_buf.rewards, dtype=np.float32),
            np.array(ppo_buf.probs, dtype=np.float32),
            np.array(ppo_buf.next_states, dtype=np.float32),
            np.array(ppo_buf.dones, dtype=np.bool_)
        )

    def get_action(self, state):
        """根据策略网络选择动作"""
        state = np.expand_dims(state, axis=0).astype(np.float32)

        if self.is_discrete:
            logits = self.get_policy(state)  # [batch, action_dim]
            dist = tfp.distributions.Categorical(logits=logits)
            action = dist.sample()[0].numpy()
            action_prob = dist.prob(action).numpy()
            return action, action_prob
        else:
            mean, log_std = self.get_policy(state)
            dist = tfp.distributions.Normal(loc=mean, scale=tf.exp(log_std))
            action = dist.sample()[0].numpy()
            action_prob = dist.prob(action).numpy()
            return action, action_prob
        
    def rollout_and_train(self,env ):
        for epoch in range(20):
            # === 批量收集数据 ===
            all_states, all_actions, all_rewards, all_log_probs, all_values, all_dones = self.collect_trajectories(env)

            # 對每條 trajectory 分別計算 advantage 和 returns
            all_adv = []
            all_returns = []

            for rewards, values, dones in zip(all_rewards, all_values, all_dones):
                adv, ret = self.compute_gae(rewards, values, dones)
                all_adv.append(adv)
                all_returns.append(ret)

            # 拼接成大 batch
            states_batch = np.concatenate(all_states, axis=0)
            actions_batch = np.concatenate(all_actions, axis=0)
            log_probs_batch = np.concatenate(all_log_probs, axis=0)
            adv_batch = np.concatenate(all_adv, axis=0)
            returns_batch = np.concatenate(all_returns, axis=0)
            # === 更新策略和价值 ===
            actor_loss,critic_loss = self.train_step(states_batch, actions_batch,adv_batch, log_probs_batch, returns_batch)


            print(f"Epoch {epoch}: actor_loss={actor_loss:.3f}, critic_loss={critic_loss:.2f}")

    def gauss_log_prob(self,mu, log_std, actions):
        # mu, log_std, actions: [batch, action_dim]
        pre_sum = -0.5 * (((actions - mu) / tf.exp(log_std)) ** 2 + 2 * log_std + np.log(2 * np.pi))
        return tf.reduce_sum(pre_sum, axis=1)  # [batch], 每个样本一个 scalar log_prob

    def train_step(self, states, actions, advantages, old_probs, returns, clip_ratio=0.1, use_ewc=False):
        """
        完整稳定版 PPO 训练步骤（支持连续动作可训练 log_std）
        """
        states = tf.convert_to_tensor(states, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.float32 if not self.is_discrete else tf.int32)
        advantages = tf.convert_to_tensor(advantages, dtype=tf.float32)
        old_probs = tf.convert_to_tensor(old_probs, dtype=tf.float32)
        returns = tf.convert_to_tensor(returns, dtype=tf.float32)

        # === 标准化 advantages ===
        advantages = (advantages - tf.reduce_mean(advantages)) / (tf.math.reduce_std(advantages) + 1e-8)

        # === Actor 更新 ===
        with tf.GradientTape() as tape:
            if self.is_discrete:
                logits = self.actor(states)
                dist = tfp.distributions.Categorical(logits=logits)
                new_log_probs = dist.log_prob(actions)
                old_log_probs = old_probs
            else:
                # 连续动作
                means = self.actor(states)  # [batch, action_dim]
                log_stds = tf.broadcast_to(self.log_std, tf.shape(means))  # 可训练 log_std
                log_stds = tf.clip_by_value(log_stds, -20.0, 2.0)  # 防止 NaN

                dist = tfp.distributions.Normal(loc=means, scale=tf.exp(log_stds))
                new_log_probs = tf.reduce_sum(dist.log_prob(actions), axis=-1)

                # 旧动作 log_prob
                old_means = old_probs
                old_log_stds = tf.broadcast_to(self.log_std, tf.shape(old_means))
                old_log_stds = tf.clip_by_value(old_log_stds, -20.0, 2.0)
                old_dist = tfp.distributions.Normal(loc=old_means, scale=tf.exp(old_log_stds))
                old_log_probs = tf.reduce_sum(old_dist.log_prob(actions), axis=-1)

            # PPO ratio
            log_ratio = new_log_probs - old_log_probs
            log_ratio = tf.clip_by_value(log_ratio, -5.0, 5.0)  # 防止 exp 溢出
            ratio = tf.exp(log_ratio)
            clipped_ratio = tf.clip_by_value(ratio, 1 - clip_ratio, 1 + clip_ratio)

            # Actor 损失
            actor_loss = -tf.reduce_mean(tf.minimum(ratio * advantages, clipped_ratio * advantages))

            # 熵奖励
            entropy = tf.reduce_mean(dist.entropy())

            # Actor 总损失
            actor_total_loss = actor_loss - self.ent_coef * entropy
            if use_ewc and self.online_fisher is not None:
                actor_total_loss += self.ewc_regularization()

        # Actor 梯度更新（包含 log_std）
        actor_vars = self.actor.trainable_variables + [self.log_std]
        actor_grads = tape.gradient(actor_total_loss, actor_vars)
        actor_grads, _ = tf.clip_by_global_norm(actor_grads, 0.3)
        self.actor_opt.apply_gradients(zip(actor_grads, actor_vars))

        # === Critic 更新 ===
        with tf.GradientTape() as tape:
            values_pred = tf.squeeze(self.critic(states))

            # 对 returns 做归一化
            returns_norm = (returns - tf.reduce_mean(returns)) / (tf.math.reduce_std(returns) + 1e-8)

            critic_loss = tf.reduce_mean(tf.square(returns_norm - values_pred))
            if use_ewc and self.online_fisher is not None:
                critic_loss += self.ewc_regularization()

        critic_grads = tape.gradient(critic_loss, self.critic.trainable_variables)
        critic_grads, _ = tf.clip_by_global_norm(critic_grads, 0.3)
        self.critic_opt.apply_gradients(zip(critic_grads, self.critic.trainable_variables))

        # === 日志打印 ===
        print(f"Actor Loss: {actor_loss.numpy():.3f}, "
              f"Critic Loss: {critic_loss.numpy():.3f}, "
              f"Entropy: {entropy.numpy():.3f}, "
              f"Returns range: {tf.reduce_min(returns).numpy():.3f} ~ {tf.reduce_max(returns).numpy():.3f}, "
              f"log_std: {tf.reduce_mean(self.log_std).numpy():.3f}")

        return actor_loss.numpy(), critic_loss.numpy()

    def _normalize_rewards(self, rewards):
        """奖励标准化"""
        rewards = np.array(rewards)
        if len(rewards) > 1:
            rewards = (rewards - np.mean(rewards)) / (np.std(rewards) + 1e-8)
        return np.clip(rewards, -10, 10)  # 限制奖励范围

    
    def _compute_advantages(self, rewards, values, next_values, dones, gamma=0.99, lam=0.95):
        """
        计算 Generalized Advantage Estimation (GAE)
        参数:
            rewards: [batch] 当前 step 的奖励
            values: [batch] 当前状态的 V(s)
            next_values: [batch] 下一状态的 V(s')
            dones: [batch] 终止标志 (1 表示结束, 0 表示继续)
            gamma: 折扣因子
            lam: GAE 衰减因子 (λ)
        返回:
            advantages: [batch] GAE 估计的优势函数
        """
        rewards = tf.cast(rewards, tf.float32)
        values = tf.cast(values, tf.float32)
        next_values = tf.cast(next_values, tf.float32)
        dones = tf.cast(dones, tf.float32)

        # δ_t = r_t + γ V(s_{t+1}) (1 - done) - V(s_t)
        deltas = rewards + gamma * next_values * (1.0 - dones) - values

        # GAE 累积计算
        advantages = []
        gae = 0.0
        for delta, done in zip(deltas[::-1], dones[::-1]):  # 从后往前遍历
            gae = delta + gamma * lam * (1.0 - done) * gae
            advantages.insert(0, gae)  # 逆序插入

        return tf.convert_to_tensor(advantages, dtype=tf.float32)

    
    
    
    
    # ------------------ EWC online 更新 ------------------
    def update_online_fisher(self, obs: tf.Tensor, action: tf.Tensor):
        self.update_counter += 1
        if self.update_counter % self.fisher_update_frequency != 0:
            return

        if self.online_fisher is None:
            self._initialize_online_fisher()

        with tf.GradientTape() as tape:
            dist = self.get_policy(obs)
            log_prob = dist.log_prob(action)
            log_prob_mean = tf.reduce_mean(log_prob)

        grads = tape.gradient(log_prob_mean, self.actor.trainable_variables)
        for var, grad in zip(self.actor.trainable_variables, grads):
            if grad is not None:
                var_name = var.name
                grad_sq = tf.square(grad)
                old_fisher = self.online_fisher.get(var_name, tf.zeros_like(var))
                self.online_fisher[var_name] = self.ema_decay * old_fisher + (1 - self.ema_decay) * grad_sq

    def _initialize_online_fisher(self):
        self.online_fisher = {}
        for var in self.actor.trainable_variables:
            self.online_fisher[var.name] = tf.zeros_like(var)

    def ewc_regularization(self):
        current_params = {v.name: v for v in self.actor.trainable_variables + self.critic.trainable_variables}
         
        if self.online_fisher is None or self.optimal_params is None:
            return tf.constant(0.0, dtype=tf.float32)

        ewc_loss = tf.constant(0.0, dtype=tf.float32)
        for var in self.actor.trainable_variables:
            name = var.name
            fisher = self.online_fisher.get(name)
            optimal = self.optimal_params.get(name)
            current = current_params.get(name)
            if fisher is not None and optimal is not None and current is not None:
                ewc_loss += tf.reduce_sum(fisher * tf.square(current - optimal))
        return self.ewc_lambda * ewc_loss

    # ------------------ PPO update with online EWC ------------------
    def update_with_online_ewc(self, obs_batch, action_batch, advantage_batch, old_log_prob_batch):
        current_params = {var.name: var for var in self.actor.trainable_variables}

        with tf.GradientTape() as tape:
            dist = self.policy(obs_batch)
            new_log_prob = dist.log_prob(action_batch)
            ratio = tf.exp(new_log_prob - old_log_prob_batch)
            clipped_ratio = tf.clip_by_value(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
            policy_loss = -tf.reduce_mean(tf.minimum(ratio * advantage_batch,
                                                     clipped_ratio * advantage_batch))
            value_loss = tf.reduce_mean(tf.square(self.critic(obs_batch) - advantage_batch))
            entropy_loss = -tf.reduce_mean(dist.entropy())
            ewc_loss = self.ewc_regularization(current_params)
            total_loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss + ewc_loss

        grads = tape.gradient(total_loss, self.actor.trainable_variables + self.critic.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.actor.trainable_variables + self.critic.trainable_variables))

        # 更新 online Fisher
        for i in range(obs_batch.shape[0]):
            self.update_online_fisher(obs_batch[i:i + 1], action_batch[i:i + 1])

        return {
            'total_loss': total_loss.numpy(),
            'policy_loss': policy_loss.numpy(),
            'value_loss': value_loss.numpy(),
            'entropy_loss': entropy_loss.numpy(),
            'ewc_loss': ewc_loss.numpy()
        }

    def _build_policy_network(self):
        """Build the policy network"""
        inputs = tf.keras.layers.Input(shape=(self.state_dim,))
        x = tf.keras.layers.Dense(64, activation='relu')(inputs)
        x = tf.keras.layers.Dense(64, activation='relu')(x)

        # Action distribution parameters - using Dense layers
        mean = tf.keras.layers.Dense(self.action_dim)(x)
        log_std = tf.keras.layers.Dense(self.action_dim)(x)

        return tf.keras.Model(inputs=inputs, outputs=[mean, log_std])

    def learn(self, buf=None,use_ewc=True, total_timesteps=1000000):
        """Simulated training process"""
        print(f"Training for {total_timesteps} timesteps...")

        for i in range(total_timesteps // 1000):
             
            self.train_buffer_step(buf,use_ewc=True)
            if i % 100 == 0:
                print(f"Step {i * 1000}/{total_timesteps}")


class ESP32PPOPolicy(tf.keras.Model):
    """統一的策略模型，支持離散和連續動作空間"""

    def __init__(self, state_dim, action_dim, hidden_units=8, is_discrete=True):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.is_discrete = is_discrete

        # 共享的特徵提取層
        self.hidden_layers = tf.keras.Sequential([
            tf.keras.layers.Dense(hidden_units, activation='relu'),
            tf.keras.layers.Dense(hidden_units, activation='relu')
        ])

        if self.is_discrete:
            # 離散動作：輸出動作概率分佈
            self.logits_layer = tf.keras.layers.Dense(action_dim, activation=None)
        else:
            # 連續動作：輸出均值和對數標準差
            self.mean_layer = tf.keras.layers.Dense(action_dim, activation=None)
            self.log_std = tf.Variable(tf.zeros(action_dim), trainable=True)

    def call(self, states):
        hidden = self.hidden_layers(states)

        if self.is_discrete:
            logits = self.logits_layer(hidden)
            return tf.nn.softmax(logits, axis=-1)  # 返回概率分佈
        else:
            mean = self.mean_layer(hidden)
            log_std = self.log_std
            std = tf.exp(log_std)
            return mean, std  # 返回高斯分佈參數

    def get_action(self, states, deterministic=False):
        """根據策略採樣動作"""
        if self.is_discrete:
            probs = self(states)
            if deterministic:
                # 確定性策略：選擇概率最大的動作
                return tf.argmax(probs, axis=-1)
            else:
                # 隨機策略：根據概率分佈採樣
                return tf.random.categorical(tf.math.log(probs), 1)[:, 0]
        else:
            mean, std = self(states)
            if deterministic:
                # 確定性策略：直接使用均值
                return mean
            else:
                # 隨機策略：從高斯分佈中採樣
                return mean + tf.random.normal(tf.shape(mean)) * std

    def log_prob(self, states, actions):
        """計算動作的對數概率"""
        if self.is_discrete:
            probs = self(states)
            actions = tf.cast(actions, tf.int32)
            # 使用one-hot編碼計算對數概率
            action_probs = tf.reduce_sum(probs * tf.one_hot(actions, self.action_dim), axis=-1)
            return tf.math.log(action_probs + 1e-8)
        else:
            mean, std = self(states)
            # 高斯分佈的對數概率
            log_prob = -0.5 * tf.reduce_sum(tf.square((actions - mean) / std), axis=-1)
            log_prob -= 0.5 * tf.math.log(2.0 * np.pi) * tf.cast(self.action_dim, tf.float32)
            log_prob -= tf.reduce_sum(tf.math.log(std))
            return log_prob


class PPOBuffer:
    def __init__(self, state_dim, action_dim, buffer_size=512, gamma=0.99, lam=0.95):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.buffer_size = buffer_size
        self.gamma = gamma

        # 初始化缓冲区数组
        self.states = np.zeros((buffer_size, state_dim), dtype=np.float32)


        self.actions = np.zeros((buffer_size, action_dim), dtype=np.float32)

        #self.actions = np.zeros(buffer_size, dtype=np.int32)  # 存整数动作
        self.rewards = np.zeros(buffer_size, dtype=np.float32)
        self.log_probs = np.zeros(buffer_size, dtype=np.float32)
        self.values = np.zeros(buffer_size, dtype=np.float32)  # 修正：应该是1D，不是2D


        self.next_values = np.zeros(buffer_size, dtype=np.float32)  # 修正：应该是1D，不是2D

        self.next_states = np.zeros((buffer_size, state_dim), dtype=np.float32)
        self.probs = np.zeros((buffer_size, action_dim), dtype=np.float32)
        self.dones = np.zeros(buffer_size, dtype=bool)  # 修正：使用bool类型

        # 用于计算returns
        self.returns = np.zeros(buffer_size, dtype=np.float32)

        self.ptr = 0
        self.num_samples = 0
 
   
        self.size = buffer_size 
        self.lam = lam

        #self.reset()

    def reset(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
        self.values = []
        self.next_values = []
        self.next_states = []

    def compute_advantages(self, rewards, values, dones, gamma=0.99, lam=0.95):
        """
        计算 Generalized Advantage Estimation (GAE-Lambda)
        Args:
            rewards: [T] 轨迹中的奖励
            values:  [T] critic 估计的 V(s_t)
            dones:   [T] 每一步是否 episode 结束
            gamma:   折扣因子
            lam:     GAE 的 λ 系数

        Returns:
            advantages: [T] 每一步的优势估计
            returns:    [T] TD(λ) 回报 (等价于优势+value)
        """
        T = len(rewards)
        values = np.array(values, dtype=np.float32)
        next_values = np.zeros_like(values)

        # 构建 next_values
        for t in range(T - 1):
            next_values[t] = values[t + 1]
        next_values[-1] = 0.0 if dones[-1] else values[-1]

        advantages = np.zeros_like(rewards, dtype=np.float32)
        gae = 0.0

        # 反向计算 GAE
        for t in reversed(range(T)):
            # δ_t = r_t + γ V(s_{t+1}) - V(s_t)
            delta = rewards[t] + gamma * next_values[t] * (1 - dones[t]) - values[t]
            gae = delta + gamma * lam * (1 - dones[t]) * gae
            advantages[t] = gae

        returns = advantages + values
        return advantages, returns

    def get_ppo_batch(self, batch_size=64, shuffle=True, gamma=0.99, lam=0.95):
        """生成 PPO 的 mini-batch，自动计算好 advantages & returns"""
        if len(self.states) == 0:
            raise ValueError("Buffer is empty!")

        # === 转 numpy ===
        states = np.array(self.states, dtype=np.float32)
        actions = np.array(self.actions, dtype=np.float32)
        log_probs = np.array(self.log_probs, dtype=np.float32)
        values = np.array(self.values, dtype=np.float32)
        rewards = np.array(self.rewards, dtype=np.float32)
        dones = np.array(self.dones, dtype=np.float32)

        # === 调用 GAE ===
        advantages, returns = self.compute_advantages(
            rewards, values, dones, gamma, lam
        )

        # === 打乱索引 ===
        n = len(states)
        indices = np.arange(n)
        if shuffle:
            np.random.shuffle(indices)

        # === mini-batch 生成器 ===
        for start in range(0, n, batch_size):
            end = start + batch_size
            batch_idx = indices[start:end]

            yield (
                states[batch_idx],  # (B, state_dim)
                actions[batch_idx],  # (B,)
                log_probs[batch_idx],  # (B,)
                returns[batch_idx],  # (B,)
                advantages[batch_idx]  # (B,)
            )

    def is_full(self):
        return self.num_samples == self.buffer_size

    def finish_path(self, last_value=0):
        """计算 GAE 或 Returns，这里简单使用 returns = rewards + last_value"""
        # 确保有数据
        if self.num_samples == 0:
            return

        self.returns = np.zeros(self.num_samples, dtype=np.float32)
        running_return = last_value

        # 反向计算returns
        for t in reversed(range(self.num_samples)):
            if self.dones[t]:
                running_return = 0
            running_return = self.rewards[t] + self.gamma * running_return
            self.returns[t] = running_return

    def clear(self):
        """清空缓冲区"""
        self.ptr = 0
        self.num_samples = 0
        # 可选：重置数组为零
        self.states.fill(0)
        self.actions.fill(0)
        self.rewards.fill(0)
        self.values.fill(0)
        self.next_states.fill(0)
        self.probs.fill(0)
        self.dones.fill(False)

    def store(self, state, action, prob, reward, next_state, done, value):
        idx = self.ptr % self.buffer_size
        self.states[idx] = state
        self.actions[idx] = action
        self.log_probs[idx] = prob
        self.rewards[idx] = reward
        self.next_states[idx] = next_state
        self.dones[idx] = done
        self.values[idx] = value

        self.ptr += 1
        self.num_samples = min(self.num_samples + 1, self.buffer_size)   

    def _safe_ensure_state_shape(self, state):
        """安全的状态形状处理"""
        if isinstance(state, (int, float, np.number)):
            # 单个数值，转换为1D数组
            return np.array([state], dtype=np.float32)
        elif hasattr(state, '__len__'):
            # 有长度的对象（数组、列表等）
            try:
                state_array = np.array(state, dtype=np.float32).flatten()
                return state_array
            except:
                return np.array([state], dtype=np.float32)
        else:
            # 其他情况，尝试转换
            return np.array([state], dtype=np.float32)

    def _safe_ensure_prob_shape(self, prob):
        """安全的概率形状处理"""
        # 检查是否是标量
        if isinstance(prob, (int, float, np.number)):
            # 单个概率值，转换为均匀分布
            prob_value = float(prob)
            if prob_value <= 0 or prob_value > 1:
                # 无效概率，使用均匀分布
                return np.full(self.action_dim, 1.0 / self.action_dim)
            else:
                # 创建基于该概率的分布（第一个动作概率高，其他均匀）
                probs = np.full(self.action_dim, (1.0 - prob_value) / (self.action_dim - 1))
                probs[0] = prob_value
                return probs
        else:
            # 尝试处理为数组
            try:
                prob_array = np.array(prob, dtype=np.float32).flatten()
                if len(prob_array) == self.action_dim:
                    return prob_array
                else:
                    # 维度不匹配，使用均匀分布
                    return np.full(self.action_dim, 1.0 / self.action_dim)
            except:
                # 转换失败，使用均匀分布
                return np.full(self.action_dim, 1.0 / self.action_dim)

    def _safe_ensure_scalar(self, value):
        """确保值是标量"""
        if isinstance(value, (int, float, np.number)):
            return float(value)
        elif hasattr(value, '__len__'):
            try:
                if len(value) > 0:
                    return float(value[0])
                else:
                    return 0.0
            except:
                return 0.0
        else:
            try:
                return float(value)
            except:
                return 0.0


    def _create_valid_probability(self, original_prob):
        """创建有效的概率分布"""
        original_prob = np.array(original_prob, dtype=np.float32).flatten()

        if len(original_prob) == 0:
            # 如果没有概率信息，返回均匀分布
            return np.full(self.action_dim, 1.0 / self.action_dim)

        # 取前action_dim个元素，不足则填充
        if len(original_prob) >= self.action_dim:
            prob = original_prob[:self.action_dim]
        else:
            # 填充剩余部分为0
            prob = np.pad(original_prob, (0, self.action_dim - len(original_prob)))

        # 归一化
        prob_sum = np.sum(prob)
        if prob_sum <= 0:
            return np.full(self.action_dim, 1.0 / self.action_dim)

        return prob / prob_sum

    def get_buffer_items(self):
        """返回 numpy 格式的完整batch"""
        return (
            np.array(self.states[:self.num_samples], dtype=np.float32),
            np.array(self.actions[:self.num_samples]),
            np.array(self.rewards[:self.num_samples], dtype=np.float32),
            np.array(self.probs[:self.num_samples], dtype=np.float32),
            np.array(self.next_states[:self.num_samples], dtype=np.float32),
            np.array(self.dones[:self.num_samples], dtype=bool)
        )

    
    def is_ready(self, batch_size):
        """检查是否有足够的数据进行训练"""
        return self.num_samples >= batch_size

    # 添加finish_trajectory方法作为finish_path的别名（为了兼容性）
    def finish_trajectory(self, last_value=0):
        """finish_path的别名方法"""
        self.finish_path(last_value)


class PPOAgent(ESP32PPOAgent):
    def __init__(self, fisher_matrix=None, optimal_params=None, state_dim=5, action_dim=4,
                 clip_epsilon=0.2, value_coef=0.5, entropy_coef=0.01,
                 ewc_lambda=500, memory_size=1000, hidden_units=32,  # 增加到 32
                 clip_ratio=0.2, actor_lr=1e-4, critic_lr=1e-3, is_discrete=False,  # 降低actor学习率
                 gamma=0.99, lam=0.95):
        super().__init__( fisher_matrix , optimal_params , state_dim , action_dim,
                 clip_epsilon, value_coef, entropy_coef,
                 ewc_lambda, memory_size, hidden_units,
                 clip_ratio, actor_lr, critic_lr,is_discrete,
                 gamma, lam)
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_units = hidden_units
        self.clip_ratio = clip_ratio
        self.gamma = gamma
        self.lam = lam        

        # 初始化优化器
        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=actor_lr)
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=critic_lr)
        
        # 构建网络
        self.actor = self._build_actor()
        self.critic = self._build_critic()

    def act_debug(self, state):
        """带调试信息的act方法"""
        print(f"=== act方法调试 ===")
        print(f"输入状态类型: {type(state)}")
        print(f"输入状态值: {state}")

        # 状态预处理
        if not isinstance(state, np.ndarray):
            state = np.array(state, dtype=np.float32)
            print(f"转换后状态: {state}")

        if len(state.shape) == 1:
            state = state.reshape(1, -1)
            print(f"重塑后形状: {state.shape}")

        print(f"最终输入形状: {state.shape}")

        # 调用actor预测
        try:
            if self.is_discrete:
                prob = self.actor.predict(state, verbose=0)[0]
                print(f"动作概率: {prob}")
                action = np.random.choice(self.action_dim, p=prob)
                print(f"选择动作: {action}")
            else:
                # 连续动作处理
                pass

            return action

        except Exception as e:
            print(f"actor预测出错: {e}")
            return 0  # 返回默认动作



    def train_ppo(self):
        """使用包含next_states的数据进行PPO训练"""
        # 获取完整数据用于分析或监控（可选）
        states, actions, rewards, probs, next_states, dones = self.ppo_buffer.get_buffer_items()
        print(f"训练数据形状: states{states.shape}, rewards{rewards.shape}")
        
        # Mini-batch训练
        for batch_idx, batch in enumerate(self.ppo_buffer.get_batch(batch_size=64)):
            states_b, actions_b, old_probs_b, returns_b, old_values_b, next_states_b, dones_b = batch
            
            # 修正：使用batch对应的rewards，而不是完整的rewards数组
            # 从完整数据中提取对应batch的rewards
            start_idx = batch_idx * 64
            end_idx = min((batch_idx + 1) * 64, len(rewards))
            rewards_b = rewards[start_idx:end_idx]
            
            # 确保维度匹配
            if len(rewards_b) != len(states_b):
                # 如果长度不匹配，使用buffer中的rewards（如果buffer存储了batch对应的rewards）
                rewards_b = self.ppo_buffer.rewards[start_idx:end_idx]
            
            # 使用next_states计算TD目标
            next_values = self.critic.predict(next_states_b)  # 注意：应该是self.critic而不是self.critic_network
            next_values = next_values.flatten()  # 确保是一维数组
            
            # 计算TD目标（修正维度）
            td_targets = rewards_b + self.ppo_buffer.gamma * next_values * (1 - dones_b.astype(np.float32))
            
            # PPO更新步骤
            # 1. 计算优势函数（修正维度）
            advantages = returns_b - old_values_b.mean(axis=1)
            
            # 2. 更新网络
            actor_loss = self.update_actor(states_b, actions_b, old_probs_b, advantages)
            critic_loss = self.update_critic(states_b, td_targets)
            
            if batch_idx % 10 == 0:
                print(f"Batch {batch_idx}: Actor Loss: {actor_loss:.4f}, Critic Loss: {critic_loss:.4f}")

    
    def update_critic(self, states, td_targets):
            """更新Critic网络"""
            with tf.GradientTape() as tape:
                # Critic预测的价值
                values = self.critic(states)
                values = tf.squeeze(values)  # 去除多余的维度
                
                # Critic损失（MSE）
                critic_loss = tf.reduce_mean(tf.square(td_targets - values))
            
            # 计算并应用梯度
            critic_gradients = tape.gradient(critic_loss, self.critic.trainable_variables)
            self.optimizer.apply_gradients(zip(critic_gradients, self.critic.trainable_variables))
            
            return critic_loss.numpy()


    def update_actor(self, states, actions, old_probs, advantages):
        with tf.GradientTape() as tape:
            # 获取新策略的概率分布 [batch_size, action_dim]
            new_probs = self.actor(states)
            print(f"Debug - new_probs shape: {new_probs.shape}")  # 应该是 [10,4]
            print(f"Debug - old_probs shape: {old_probs.shape}")  # 应该是 [10,4]
            print(f"Debug - actions shape: {actions.shape}")      # 应该是 [10]
            print(f"Debug - advantages shape: {advantages.shape}") # 应该是 [10]
            
            # 确保动作是整数类型
            actions = tf.cast(actions, tf.int32)
            
            # 方法1：使用 tf.gather 选择对应动作的概率
            # 创建批次索引
            batch_size = tf.shape(states)[0]
            batch_indices = tf.range(batch_size)
            
            # 选择每个样本对应动作的概率
            # new_probs[batch_indices, actions] 会选择每个样本对应动作的概率
            new_probs_selected = tf.gather(new_probs, actions, batch_dims=1)
            old_probs_selected = tf.gather(old_probs, actions, batch_dims=1)
            
            print(f"Debug - new_probs_selected shape: {new_probs_selected.shape}")  # 应该是 [10]
            print(f"Debug - old_probs_selected shape: {old_probs_selected.shape}")  # 应该是 [10]
            
            # 计算概率比 [batch_size]
            ratio = new_probs_selected / (old_probs_selected + 1e-8)
            print(f"Debug - ratio shape: {ratio.shape}")  # 应该是 [10]
            
            # 确保advantages形状正确 [batch_size]
            advantages = tf.reshape(advantages, [-1])
            
            # 现在可以相乘了，因为都是 [batch_size] 形状
            surr1 = ratio * advantages
            surr2 = tf.clip_by_value(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages
            
            # PPO损失函数
            actor_loss = -tf.reduce_mean(tf.minimum(surr1, surr2))
        
        # 计算梯度并更新
        actor_grads = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))
        
        return actor_loss

    def save_task_parameters(self, task_id):
        """
        保存当前任务的参数用于EWC持续学习
        """
        print(f"保存任务 {task_id} 的参数用于EWC...")

        # 保存当前参数值（均值）
        self.ewc_means[task_id] = {}
        for var in self.actor.trainable_variables + self.critic.trainable_variables:
            self.ewc_means[task_id][var.name] = var.numpy().copy()

        # 计算Fisher信息矩阵（参数重要性）
        self.ewc_fisher[task_id] = self.compute_fisher_matrix()

        self.ewc_task_count += 1

        # 保存到文件
        self._save_ewc_to_file(task_id)

    def compute_fisher_matrix(self, num_samples=100):
        """
        计算Fisher信息矩阵，估计参数的重要性
        """
        fisher = {}

        # 初始化Fisher矩阵
        for var in self.actor.trainable_variables + self.critic.trainable_variables:
            fisher[var.name] = np.zeros_like(var.numpy())

        # 通过采样计算梯度平方的期望
        # 这里简化实现，实际应用中需要根据具体任务调整
        for _ in range(num_samples):
            # 生成随机输入
            random_state = tf.random.normal((1, self.state_dim))

            with tf.GradientTape() as tape:
                # 计算actor输出
                action_probs = self.actor(random_state)
                # 计算对数概率
                log_probs = tf.math.log(action_probs + 1e-8)
                # 计算损失（这里使用负熵作为示例）
                loss = -tf.reduce_sum(action_probs * log_probs)

            # 计算梯度
            gradients = tape.gradient(loss, self.actor.trainable_variables + self.critic.trainable_variables)

            # 累加梯度平方
            for var, grad in zip(self.actor.trainable_variables + self.critic.trainable_variables, gradients):
                if grad is not None:
                    fisher[var.name] += (grad.numpy() ** 2) / num_samples

        return fisher



    def _save_ewc_to_file(self, task_id):
        """保存EWC参数到文件 - 使用NumPy格式"""
        # 确保目录存在
        os.makedirs('ewc_params', exist_ok=True)

        # 保存means
        means_dir = f'ewc_params/task_{task_id}_means'
        os.makedirs(means_dir, exist_ok=True)

        for var_name, mean_value in self.ewc_means[task_id].items():
            # 清理变量名，使其适合作为文件名
            safe_name = var_name.replace('/', '_').replace(':', '_')
            np.save(f'{means_dir}/{safe_name}.npy', mean_value)

        # 保存fisher信息
        fisher_dir = f'ewc_params/task_{task_id}_fisher'
        os.makedirs(fisher_dir, exist_ok=True)

        for var_name, fisher_value in self.ewc_fisher[task_id].items():
            safe_name = var_name.replace('/', '_').replace(':', '_')
            np.save(f'{fisher_dir}/{safe_name}.npy', fisher_value)

        # 保存元数据
        metadata = {
            'task_id': task_id,
            'saved_variables': list(self.ewc_means[task_id].keys()),
            'timestamp': np.datetime64('now').astype(str)
        }

        with open(f'ewc_params/task_{task_id}_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"任务 {task_id} 的EWC参数已保存")

    def load_ewc_parameters(self, task_id):
        """从文件加载EWC参数"""
        try:
            # 加载元数据
            with open(f'ewc_params/task_{task_id}_metadata.json', 'r') as f:
                metadata = json.load(f)

            # 初始化存储
            self.ewc_means[task_id] = {}
            self.ewc_fisher[task_id] = {}

            # 加载means
            means_dir = f'ewc_params/task_{task_id}_means'
            for var_name in metadata['saved_variables']:
                safe_name = var_name.replace('/', '_').replace(':', '_')
                mean_value = np.load(f'{means_dir}/{safe_name}.npy')
                self.ewc_means[task_id][var_name] = mean_value

            # 加载fisher
            fisher_dir = f'ewc_params/task_{task_id}_fisher'
            for var_name in metadata['saved_variables']:
                safe_name = var_name.replace('/', '_').replace(':', '_')
                fisher_value = np.load(f'{fisher_dir}/{safe_name}.npy')
                self.ewc_fisher[task_id][var_name] = fisher_value

            print(f"任务 {task_id} 的EWC参数已加载")
            return True

        except FileNotFoundError as e:
            print(f"加载任务 {task_id} 的EWC参数失败: {e}")
            return False

    def ewc_regularization_loss(self):
        """
        计算EWC正则化损失，防止灾难性遗忘
        """
        if self.ewc_task_count == 0:
            return 0.0

        ewc_loss = 0.0

        for task_id in range(self.ewc_task_count):
            for var in self.actor.trainable_variables + self.critic.trainable_variables:
                if var.name in self.ewc_means[task_id] and var.name in self.ewc_fisher[task_id]:
                    # EWC损失：Fisher * (当前参数 - 旧参数)^2
                    mean = self.ewc_means[task_id][var.name]
                    fisher = self.ewc_fisher[task_id][var.name]
                    ewc_loss += tf.reduce_sum(fisher * (var - mean) ** 2)

        return self.ewc_lambda * ewc_loss

    def collect_and_train(self, env, num_episodes=100):
        """简化的收集和训练方法"""
        for episode in range(num_episodes):
            state = env.reset()
            done = False
            episode_reward = 0

            while not done:
                action, action_prob, value = self.select_action(state)
                next_state, reward, done, _ = env.step(action)

                # 这里应该将经验存入buffer，然后采样训练
                # 简化实现：直接进行在线学习

                state = next_state
                episode_reward += reward

            if episode % 10 == 0:
                print(f"回合 {episode}, 奖励: {episode_reward}")

    def xselect_action(self, state):
        """选择动作"""
        state_tensor = tf.convert_to_tensor([state], dtype=tf.float32)
        action_probs = self.actor(state_tensor)
        action = tf.random.categorical(tf.math.log(action_probs), 1)[0, 0]
        value = self.critic(state_tensor)
        return action.numpy(), action_probs[0].numpy(), value[0, 0].numpy()

    def _continuous_action_training_with_next_state(self, states, actions, advantages, old_probs, returns, next_states):
        """處理連續動作空間的訓練（包含next_state）"""
        with tf.GradientTape(persistent=True) as tape:
            # 獲取策略輸出 (均值)
            means = self.policy(states)  # shape: [batch_size, action_dim]

            # 使用可訓練的log_std
            log_stds = tf.ones_like(means) * self.log_std

            # 創建正態分布
            dist = tfp.distributions.Normal(loc=means, scale=tf.exp(log_stds))

            # 計算新動作的概率 (對數概率)
            new_log_probs = dist.log_prob(actions)  # shape: [batch_size, action_dim]
            new_log_probs = tf.reduce_sum(new_log_probs, axis=1, keepdims=True)  # shape: [batch_size, 1]

            # 處理舊概率
            if len(old_probs.shape) == 1:
                old_log_probs = tf.reshape(old_probs, [-1, 1])  # shape: [batch_size, 1]
            else:
                old_log_probs = old_probs

            # 確保形狀匹配
            if old_log_probs.shape != new_log_probs.shape:
                print(f"形狀不匹配: old_log_probs {old_log_probs.shape}, new_log_probs {new_log_probs.shape}")
                # 使用默認值
                old_log_probs = tf.ones_like(new_log_probs) * -1.0  # 默認對數概率

            # 計算比率 (使用對數概率)
            ratio = tf.exp(new_log_probs - old_log_probs)
            print(f"比率形狀: {ratio.shape}, 範圍: [{tf.reduce_min(ratio):.3f}, {tf.reduce_max(ratio):.3f}]")

            # 調整advantages形狀
            advantages_reshaped = tf.reshape(advantages, [-1, 1])

            # PPO裁剪損失
            surr1 = ratio * advantages_reshaped
            surr2 = tf.clip_by_value(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages_reshaped
            actor_loss = -tf.reduce_mean(tf.minimum(surr1, surr2))

            # 熵正則化
            entropy = tf.reduce_mean(dist.entropy())

            # Critic損失 - 使用next_states計算更好的價值估計
            current_values = self.critic(states)  # shape: [batch_size, 1]
            next_values = self.critic(next_states)  # shape: [batch_size, 1]

            # 使用TD誤差計算更好的目標值
            targets = returns  # 或者使用: advantages + tf.squeeze(current_values)
            critic_loss = tf.reduce_mean(tf.square(targets - tf.squeeze(current_values)))

            # 總損失
            total_loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy

        # 分別計算梯度
        actor_variables = self.policy.trainable_variables + [self.log_std]
        critic_variables = self.critic.trainable_variables

        actor_grads = tape.gradient(total_loss, actor_variables)
        critic_grads = tape.gradient(critic_loss, critic_variables)

        # 梯度裁剪
        if actor_grads is not None:
            actor_grads, _ = tf.clip_by_global_norm(actor_grads, 0.5)
            self.actor_optimizer.apply_gradients(zip(actor_grads, actor_variables))

        if critic_grads is not None:
            critic_grads, _ = tf.clip_by_global_norm(critic_grads, 0.5)
            self.critic_optimizer.apply_gradients(zip(critic_grads, critic_variables))

        del tape  # 釋放持久tape

        print(f"回放完成 - Actor損失: {actor_loss:.4f}, Critic損失: {critic_loss:.4f}, 熵: {entropy:.4f}")


    def _discrete_action_training(self, states, actions, advantages, old_probs, returns, next_states):
        """處理離散動作空間的訓練"""
        with tf.GradientTape(persistent=True) as tape:
            # 獲取策略輸出 (logits)
            logits = self.policy(states)  # shape: [batch_size, action_dim]

            # 創建分類分布
            dist = tfp.distributions.Categorical(logits=logits)

            # 計算新動作的概率
            new_log_probs = dist.log_prob(actions)  # shape: [batch_size]
            new_log_probs = tf.reshape(new_log_probs, [-1, 1])  # shape: [batch_size, 1]

            # 處理舊概率
            if len(old_probs.shape) == 1:
                old_log_probs = tf.reshape(old_probs, [-1, 1])
            else:
                old_log_probs = old_probs

            # 確保形狀匹配
            if old_log_probs.shape != new_log_probs.shape:
                print(f"形狀不匹配: old_log_probs {old_log_probs.shape}, new_log_probs {new_log_probs.shape}")
                old_log_probs = tf.ones_like(new_log_probs) * -1.0

            # 計算比率
            ratio = tf.exp(new_log_probs - old_log_probs)
            print(f"離散動作 - 比率形狀: {ratio.shape}, 範圍: [{tf.reduce_min(ratio):.3f}, {tf.reduce_max(ratio):.3f}]")

            # 調整advantages形狀
            advantages_reshaped = tf.reshape(advantages, [-1, 1])

            # PPO裁剪損失
            surr1 = ratio * advantages_reshaped
            surr2 = tf.clip_by_value(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages_reshaped
            actor_loss = -tf.reduce_mean(tf.minimum(surr1, surr2))

            # 熵正則化
            entropy = tf.reduce_mean(dist.entropy())

            # Critic損失
            current_values = self.critic(states)
            next_values = self.critic(next_states)
            targets = returns
            critic_loss = tf.reduce_mean(tf.square(targets - tf.squeeze(current_values)))

            # 總損失
            total_loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy

        # 分別計算梯度
        actor_variables = self.policy.trainable_variables
        critic_variables = self.critic.trainable_variables

        actor_grads = tape.gradient(total_loss, actor_variables)
        critic_grads = tape.gradient(critic_loss, critic_variables)

        # 梯度裁剪
        if actor_grads is not None:
            actor_grads, _ = tf.clip_by_global_norm(actor_grads, 0.5)
            self.actor_optimizer.apply_gradients(zip(actor_grads, actor_variables))

        if critic_grads is not None:
            critic_grads, _ = tf.clip_by_global_norm(critic_grads, 0.5)
            self.critic_optimizer.apply_gradients(zip(critic_grads, critic_variables))

        del tape

        print(f"離散動作訓練 - Actor損失: {actor_loss:.4f}, Critic損失: {critic_loss:.4f}, 熵: {entropy:.4f}")

    def _continuous_action_training(self, states, actions, advantages, old_probs, returns, next_states):
        """處理連續動作空間的訓練"""
        with tf.GradientTape(persistent=True) as tape:
            # 獲取策略輸出 (均值)
            means = self.policy(states)
            log_stds = tf.ones_like(means) * self.log_std
            dist = tfp.distributions.Normal(loc=means, scale=tf.exp(log_stds))

            # 計算新動作的概率
            new_log_probs = dist.log_prob(actions)
            new_log_probs = tf.reduce_sum(new_log_probs, axis=1, keepdims=True)

            # 處理舊概率
            if len(old_probs.shape) == 1:
                old_log_probs = tf.reshape(old_probs, [-1, 1])
            else:
                old_log_probs = old_probs

            if old_log_probs.shape != new_log_probs.shape:
                print(f"形狀不匹配: old_log_probs {old_log_probs.shape}, new_log_probs {new_log_probs.shape}")
                old_log_probs = tf.ones_like(new_log_probs) * -1.0

            # 計算比率
            ratio = tf.exp(new_log_probs - old_log_probs)
            print(f"連續動作 - 比率形狀: {ratio.shape}, 範圍: [{tf.reduce_min(ratio):.3f}, {tf.reduce_max(ratio):.3f}]")

            advantages_reshaped = tf.reshape(advantages, [-1, 1])

            # PPO裁剪損失
            surr1 = ratio * advantages_reshaped
            surr2 = tf.clip_by_value(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages_reshaped
            actor_loss = -tf.reduce_mean(tf.minimum(surr1, surr2))

            # 熵正則化
            entropy = tf.reduce_mean(dist.entropy())

            # Critic損失
            current_values = self.critic(states)
            targets = returns
            critic_loss = tf.reduce_mean(tf.square(targets - tf.squeeze(current_values)))

            total_loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy

        # 分別計算梯度
        actor_variables = self.policy.trainable_variables + [self.log_std]
        critic_variables = self.critic.trainable_variables

        actor_grads = tape.gradient(total_loss, actor_variables)
        critic_grads = tape.gradient(critic_loss, critic_variables)

        if actor_grads is not None:
            actor_grads, _ = tf.clip_by_global_norm(actor_grads, 0.5)
            self.actor_optimizer.apply_gradients(zip(actor_grads, actor_variables))

        if critic_grads is not None:
            critic_grads, _ = tf.clip_by_global_norm(critic_grads, 0.5)
            self.critic_optimizer.apply_gradients(zip(critic_grads, critic_variables))

        del tape

        print(f"連續動作訓練 - Actor損失: {actor_loss:.4f}, Critic損失: {critic_loss:.4f}, 熵: {entropy:.4f}")




