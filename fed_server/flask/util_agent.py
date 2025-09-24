import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np
import tensorflow_probability as tfp

import datetime
from collections import deque
import os

from util_env import PlantLLLHVACEnv
tfd = tfp.distributions

# state_dim = 5  # 健康狀態、溫度、濕度、光照、CO2
# action_dim = 4  # 4個控制動作
class PPOBaseAgent:
    def __init__(self, state_dim=5, action_dim=4,
                 clip_epsilon=0.2, value_coef=0.5, entropy_coef=0.01,
                 ewc_lambda=500, memory_size=1000):
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.ewc_lambda = ewc_lambda  # EWC正则化强度
        self.env=PlantLLLHVACEnv()
        # 构建网络
        self.actor = self._build_actor()
        self.critic = self._build_critic()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=3e-4)

        # 持续学习相关
        self.task_memory = deque(maxlen=memory_size)  # 存储之前任务的经验
        self.fisher_matrices = {}  # 存储每个任务的Fisher信息矩阵
        self.optimal_params = {}  # 存储每个任务的最优参数
        self.current_task_id = 0


    def _build_actor(self):
        return tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(self.state_dim,)),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(self.action_dim, activation='sigmoid')
        ])


    def _build_critic(self):
        return tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(self.state_dim,)),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(1, activation='linear')
        ])

    def select_action(self, state):
        state = tf.expand_dims(tf.convert_to_tensor(state, dtype=tf.float32), 0)
        probs = self.actor(state)
        action = tf.cast(probs > 0.5, tf.float32)  # 二值化动作
        return action.numpy()[0], probs.numpy()[0]

    def get_action(self, state,return_probs=False):
        """Get action for environment interaction (same as select_action)"""

        action, probs = self.select_action(state)
        if return_probs:
            return action,probs
        else:
            return action

    def get_value(self, state):
        state_tensor = np.expand_dims(state, axis=0)  # (1, state_dim)
        value = self.critic(state_tensor).numpy()[0, 0]
        return value

    def _count_params(self, model):
        """計算模型參數數量"""
        return sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])

    def _compute_fisher_matrix(self, experiences):
        
        states, actions, _, old_probs, _ = experiences
        # 确保 states/actions 是 tf.Tensor
        states = tf.convert_to_tensor(states, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.float32)

        trainable_vars = self.actor.trainable_variables + self.critic.trainable_variables
        # 初始化 fisher 为与每个 var 同形的 0 张量
        fisher_accum = [tf.zeros_like(var, dtype=tf.float32) for var in trainable_vars]
        n = tf.cast(tf.shape(states)[0], tf.float32)

        # 逐样本或小 batch 处理以节省内存（这里按整体计算）
        with tf.GradientTape() as tape:
            # 计算对数概率：对于独立的 Bernoulli 每维相乘 -> 对数相加
            new_probs = self.actor(states, training=False)  # shape [B, action_dim]
            actions_float = tf.cast(actions, tf.float32)
            single_sample_probs = tf.reduce_sum(new_probs * actions_float + (1. - new_probs) * (1. - actions_float), axis=1)  # [B]
            log_probs = tf.math.log(single_sample_probs + 1e-8)  # [B]
            # 为得到标量，求均值
            logp_mean = tf.reduce_mean(log_probs)

        grads = tape.gradient(logp_mean, trainable_vars)  # grads 与 vars 对应
        for i, g in enumerate(grads):
            if g is None:
                continue
            # grads 是对均值求导，相当于对每个样本梯度平均，因此 Fisher ≈ E[grad^2]
            fisher_accum[i] = tf.square(g)  # 与 var 同形

        # 返回 list，元素与 trainable_vars 顺序一致
        return fisher_accum
    
        
    def save_task_knowledge(self, task_experiences):

        if not hasattr(self, 'optimal_params') or self.optimal_params is None:
            self.optimal_params = {}

            # 确保 fisher_matrices 已初始化
        if not hasattr(self, 'fisher_matrices') or self.fisher_matrices is None:
            self.fisher_matrices = {}

            # 确保 task_memory 已初始化
        if not hasattr(self, 'task_memory') or self.task_memory is None:
            self.task_memory = []

        self.optimal_params[self.current_task_id] = [
            tf.identity(var) for var in self.actor.trainable_variables + self.critic.trainable_variables
        ]

        # 计算并保存Fisher信息矩阵
        self.fisher_matrices[self.current_task_id] = self._compute_fisher_matrix(task_experiences)

        # 保存任务经验到记忆库
        self.task_memory.extend(self._process_experiences_for_memory(task_experiences))

        print(f"任务 {self.current_task_id} 知识已保存")
        self.current_task_id += 1

    def _process_experiences_for_memory(self, experiences):
        """处理经验以便存储到长期记忆"""
        states, actions, advantages, old_probs, returns = experiences
        processed_experiences = []

        for i in range(states.shape[0]):
            processed_experiences.append((
                states[i].numpy(),
                actions[i].numpy(),
                advantages[i].numpy(),
                old_probs[i].numpy(),
                returns[i].numpy(),
                self.current_task_id  # 标记来自哪个任务
            ))

        return processed_experiences

    def replay_previous_tasks(self, batch_size=32):
        """回放先前任務的記憶"""
        if len(self.task_memory) == 0:
            return

        try:
            # 隨機採樣一批記憶
            indices = np.random.choice(len(self.task_memory),
                                       size=min(batch_size, len(self.task_memory)),
                                       replace=False)
            batch = [self.task_memory[i] for i in indices]

            print(f"回放批次大小: {len(batch)}")

            # 安全地解包數據
            states_list, actions_list, advantages_list, old_probs_list, returns_list = [], [], [], [], []

            for i, exp in enumerate(batch):
                try:
                    if len(exp) >= 5:
                        # 狀態數據處理
                        state = exp[0]
                        if isinstance(state, (list, np.ndarray)):
                            states_list.append(np.array(state, dtype=np.float32).flatten())
                        else:
                            states_list.append(np.array([state], dtype=np.float32))

                        # 動作數據處理
                        action = exp[1]
                        if isinstance(action, (list, np.ndarray)):
                            action_array = np.array(action, dtype=np.float32)
                            actions_list.append(action_array)
                        else:
                            actions_list.append(np.float32(action))

                        # 其他數據處理
                        advantages_list.append(np.float32(exp[2]))
                        old_probs_list.append(np.float32(exp[3]))
                        returns_list.append(np.float32(exp[4]))

                except Exception as e:
                    print(f"處理經驗數據 {i} 時出錯: {e}")
                    continue

            if len(states_list) == 0:
                print("警告：沒有有效的經驗數據")
                return

            # 轉換為張量
            states = tf.convert_to_tensor(states_list, dtype=tf.float32)
            actions = tf.convert_to_tensor(actions_list, dtype=tf.float32)
            advantages = tf.convert_to_tensor(advantages_list, dtype=tf.float32)
            old_probs = tf.convert_to_tensor(old_probs_list, dtype=tf.float32)
            returns = tf.convert_to_tensor(returns_list, dtype=tf.float32)

            print(f"張量形狀 - states: {states.shape}, actions: {actions.shape}")
            print(f"動作數據類型: {actions.dtype}")

            # 訓練步驟
            with tf.GradientTape() as tape:
                # 獲取新策略的輸出
                policy_output = self.policy(states)

                if self.is_discrete:
                    # 離散動作空間處理
                    new_probs = policy_output

                    # 確保動作是整型（用於one-hot編碼）
                    if actions.dtype != tf.int32:
                        actions_int = tf.cast(actions, tf.int32)
                    else:
                        actions_int = actions

                    # 處理動作形狀
                    if len(actions_int.shape) == 1:
                        actions_int = tf.reshape(actions_int, [-1, 1])

                    actions_one_hot = tf.one_hot(tf.squeeze(actions_int), depth=self.action_dim)

                    # 確保old_probs形狀正確
                    if len(old_probs.shape) == 1:
                        old_probs_reshaped = tf.reshape(old_probs, [-1, self.action_dim])
                    else:
                        old_probs_reshaped = old_probs

                    # 計算概率
                    old_action_probs = tf.reduce_sum(old_probs_reshaped * actions_one_hot, axis=1, keepdims=True)
                    new_action_probs = tf.reduce_sum(new_probs * actions_one_hot, axis=1, keepdims=True)

                else:
                    # 連續動作空間處理
                    new_mean = policy_output  # 假設policy直接輸出均值

                    # 對於連續動作，old_probs應該包含舊的均值和標準差
                    # 這裡需要根據您的實際數據格式進行調整
                    if len(old_probs.shape) == 1:
                        # 如果old_probs是標量，假設是舊的概率值
                        old_action_probs = tf.reshape(old_probs, [-1, 1])
                    else:
                        # 如果old_probs已經是概率值
                        old_action_probs = old_probs

                    # 簡單的連續動作概率計算（簡化版本）
                    # 注意：這需要根據您的實際需求調整
                    new_action_probs = tf.reduce_sum(new_mean * actions, axis=1, keepdims=True)

                    # 如果old_probs形狀不匹配，使用簡單的默認值
                    if old_action_probs.shape != new_action_probs.shape:
                        old_action_probs = tf.ones_like(new_action_probs) * 0.5  # 默認概率

                # 計算比率
                ratio = new_action_probs / (old_action_probs + 1e-8)
                print(f"比率形狀: {ratio.shape}, 數值範圍: [{tf.reduce_min(ratio):.3f}, {tf.reduce_max(ratio):.3f}]")

                # 調整advantages形狀
                advantages_reshaped = tf.reshape(advantages, ratio.shape)

                # PPO裁剪損失
                surr1 = ratio * advantages_reshaped
                surr2 = tf.clip_by_value(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages_reshaped
                actor_loss = -tf.reduce_mean(tf.minimum(surr1, surr2))

                # Critic損失
                values = self.critic(states)
                critic_loss = tf.reduce_mean(tf.square(returns - tf.reshape(values, [-1])))

                total_loss = actor_loss + 0.5 * critic_loss

            # 應用梯度
            gradients = tape.gradient(total_loss, self.policy.trainable_variables + self.critic.trainable_variables)

            if gradients is not None:
                actor_grads = gradients[:len(self.policy.trainable_variables)]
                critic_grads = gradients[len(self.policy.trainable_variables):]

                if any(g is not None for g in actor_grads):
                    self.actor_optimizer.apply_gradients(
                        zip(actor_grads, self.policy.trainable_variables)
                    )

                if any(g is not None for g in critic_grads):
                    self.critic_optimizer.apply_gradients(
                        zip(critic_grads, self.critic.trainable_variables)
                    )

            print(f"回放完成 - Actor損失: {actor_loss:.4f}, Critic損失: {critic_loss:.4f}")

        except Exception as e:
            print(f"回放過程中出現錯誤: {e}")
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


    def train_step_onehot(self, states, actions, advantages, old_probs, returns, use_ewc=True):
        """
        states: (batch_size, state_dim)
        actions: (batch_size, action_dim) one-hot
        advantages: (batch_size, action_dim)
        old_probs: (batch_size, action_dim)
        returns: (batch_size, action_dim)
        """
        with tf.GradientTape() as tape:
            new_probs = self.actor(states, training=True)        # (batch_size, action_dim)
            new_values = self.critic(states, training=True)      # (batch_size, action_dim)

            # 选择动作对应概率
            new_action_probs = tf.reduce_sum(new_probs * actions, axis=1)
            old_action_probs = tf.reduce_sum(old_probs * actions, axis=1)
            ratio = new_action_probs / (old_action_probs + 1e-8)
            clipped_ratio = tf.clip_by_value(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)

            # Advantage 只选择对应动作
            selected_advantages = tf.reduce_sum(advantages * actions, axis=1)
            policy_loss = -tf.reduce_mean(tf.minimum(ratio * selected_advantages,
                                                    clipped_ratio * selected_advantages))

            # Value loss 只选择动作对应列
            selected_values = tf.reduce_sum(new_values * actions, axis=1)
            selected_returns = tf.reduce_sum(returns * actions, axis=1)
            value_loss = tf.reduce_mean(tf.square(selected_returns - selected_values))

            entropy = -tf.reduce_mean(tf.reduce_sum(new_probs * tf.math.log(new_probs + 1e-8), axis=1))

            base_loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
            ewc_loss = self._compute_ewc_loss() if use_ewc and self.fisher_matrices else 0
            total_loss = base_loss + ewc_loss

        grads = tape.gradient(total_loss, self.actor.trainable_variables + self.critic.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.actor.trainable_variables + self.critic.trainable_variables))
        return total_loss, policy_loss, value_loss, entropy, ewc_loss

    def test_task_performance(self, env, task_id=None):
            """测试在特定任务上的性能"""
            state = env.reset()
            total_reward = 0
            done = False

            while not done:
                action, _ = self.select_action(state)
                next_state, reward, done, _ = env.step(action)
                total_reward += reward
                state = next_state

            return total_reward

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

class ESP32PPOAgent(PPOBaseAgent):
    """專為ESP32設計的輕量級PPO代理,繼承自LifelongPPOAgent"""

    def __init__(self, state_dim=5, action_dim=4,
                 clip_epsilon=0.2, value_coef=0.5, entropy_coef=0.01,
                 ewc_lambda=0.4, memory_size=1000, hidden_units=8,is_discrete=False):
        """
        初始化ESP32專用代理

        Args:
            hidden_units: 隱藏層神經元數量,根據ESP32內存調整
        """
        super().__init__(state_dim, action_dim, clip_epsilon, value_coef,
                         entropy_coef, ewc_lambda, memory_size)
        self.fisher_matrix = None
        self.optimal_params = None
        self.old_policy_params=None
        self.ewc_lambda = ewc_lambda
        self.hidden_units = hidden_units
        self._tflite_models = {}

        # 重新構建更小的網絡

        self.policy=ESP32PPOPolicy(state_dim, action_dim, hidden_units, is_discrete)
        self.critic = self._build_esp32_critic()
        self.actor = self.policy
        # 使用更小的學習率
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
        self.is_discrete=is_discrete
        print(f"ESP32代理初始化完成: {hidden_units}隱藏單元")


    def _build_esp32_critic(self):
        """構建適合ESP32的輕量級Critic網絡"""
        return tf.keras.Sequential([
            tf.keras.layers.Dense(self.hidden_units, activation='relu',
                                  input_shape=(self.state_dim,),
                                  kernel_regularizer=tf.keras.regularizers.l2(0.001)),
            tf.keras.layers.Dense(1, activation='linear',
                                  kernel_regularizer=tf.keras.regularizers.l2(0.001))
        ], name='esp32_critic')
    
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
        """
        將模型轉換為TFLite格式,針對ESP32優化
        """
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
                converter.target_spec.supported_types = [tf.float16]

            converter.target_spec.supported_ops = [
                tf.lite.OpsSet.TFLITE_BUILTINS,
                tf.lite.OpsSet.SELECT_TF_OPS
            ]

            converter.experimental_new_converter = True
            converter._experimental_lower_tensor_list_ops = False

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

    def get_action_esp32(self, state):
        """專為ESP32設計的動作獲取方法"""
        try:
            probs = self.predict_with_tflite(state, 'actor')
            action = (probs > 0.5).astype(np.float32)
            return action
        except:
            return self.get_action(state)

    def get_value_esp32(self, state):
        """專為ESP32設計的值預測方法"""
        try:
            return self.predict_with_tflite(state, 'critic')[0]
        except:
            return self.critic.predict(np.array([state]))[0][0]

    def export_for_esp32(self, base_path="ppo_model"):
        """導出ESP32所需的所有文件"""
        os.makedirs(base_path, exist_ok=True)

        # 轉換並保存TFLite模型
        self.actor=self.convert_to_tflite(model_type='actor', quantize=False, optimize_size=False)
        self.critic=self.convert_to_tflite(model_type='critic', quantize=False, optimize_size=False)

        # 使用正確的方法名
        self.save_tflite_model(f"{base_path}/ppo_policy_actor.tflite", 'actor')
        self.save_tflite_model(f"{base_path}/ppo_policy_critic.tflite", 'critic')

        # 生成C頭文件
        self._generate_c_header(f"{base_path}/ppo_policy_actor.tflite", f"{base_path}/ppo_policy_actor.h", 'actor_model')
        self._generate_c_header(f"{base_path}/ppo_policy_critic.tflite", f"{base_path}/ppo_policy_critic.h", 'critic_model')

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
            self.actor = self._build_esp32_actor()
            self.critic = self._build_esp32_critic()

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
 
    def compute_fisher_matrix(self, dataset: np.ndarray):
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

    def save_fisher_and_params(self, path: str):
        np.savez_compressed(
            path,
            fisher={k: v for k,v in self.fisher_matrix.items()},
            optimal={k: v for k,v in self.optimal_params.items()}
        )
        print(f"  Fisher matrix & optimal params saved to {path}")

    def get_policy(self, states):
        return self.policy(states)

    def  get_policy0(self, states):
        hidden = self.actor(states)

        if self.is_discrete:
            return self.logits_layer(hidden)
        else:
            mean = self.actor(states)   # 直接取输出
            log_std = self.log_std          # 可训练参数
            std = tf.exp(log_std)
            return mean, std

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



    def train_ppo_step(self, states, actions, advantages, old_probs, returns,clip_ratio=0.1, use_ewc=False):
        """执行一次 PPO 更新"""
        states = tf.convert_to_tensor(states, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.float32 if not self.is_discrete else tf.int32)
        advantages = tf.convert_to_tensor(advantages, dtype=tf.float32)
        old_probs = tf.convert_to_tensor(old_probs, dtype=tf.float32)
        returns = tf.convert_to_tensor(returns, dtype=tf.float32)

        with tf.GradientTape() as tape:
            if self.is_discrete:
                logits = self.get_policy(states)

                if self.action_dim == 1:
                    dist = tfp.distributions.Bernoulli(logits=logits)
                else:
                    dist = tfp.distributions.Categorical(logits=logits)
                log_probs = dist.log_prob(actions)
            else:
                mean, log_std = self.get_policy(states)
                std = tf.exp(log_std)
                dist = tfp.distributions.Normal(mean, std)
                log_probs = tf.reduce_sum(dist.log_prob(actions), axis=-1)
            log_probs = tf.expand_dims(log_probs, axis=-1)
            log_probs = tf.tile(log_probs, [1, 4])
            old_probs = tf.squeeze(old_probs, axis=1)
            ratio = tf.exp(log_probs - old_probs+ 1e-8)
            surrogate1 = ratio * advantages
            surrogate2 = tf.clip_by_value(ratio, 1.0 -  clip_ratio, 1.0 + clip_ratio) * advantages
            policy_loss = -tf.reduce_mean(tf.minimum(surrogate1, surrogate2))

            # Value loss
            values = self.value_net(states)
            value_loss = tf.reduce_mean((returns - tf.squeeze(values))**2)

            # 总损失
            loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * tf.reduce_mean(dist.entropy())

            if use_ewc:
                loss += self._compute_ewc_loss()

        grads = tape.gradient(loss, self.policy_params + self.value_params)
        self.optimizer.apply_gradients(zip(grads, self.policy_params + self.value_params))

        return loss


    @staticmethod
    def process_experiences(agent,experiences, gamma=0.99, gae_lambda=0.95):
        """
        处理经验并计算advantages和returns
        """
        # 提取数据
        #states = [exp[0] for exp in experiences]
        #actions = [exp[1] for exp in experiences]
        #rewards = [exp[2] for exp in experiences]
        #next_states = [exp[3] for exp in experiences]
        #dones = [exp[4] for exp in experiences]
        #old_probs = [exp[5] for exp in experiences]
        states = [exp["state"] for exp in experiences]
        actions = [exp["action"] for exp in experiences]
        rewards = [exp["reward"] for exp in experiences]
        next_states = [exp["next_state"] for exp in experiences]
        dones = [exp["done"] for exp in experiences]
        old_probs = [exp["old_prob"] for exp in experiences]
        # 转换为Tensor
        states = tf.stack(states)
        actions = tf.stack(actions)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
        dones = tf.convert_to_tensor(dones, dtype=tf.float32)
        old_probs = tf.stack(old_probs)

        # 使用critic网络计算状态价值
        with tf.GradientTape() as tape:
            values = agent.critic(states, training=True)
            next_values = agent.critic(tf.stack(next_states), training=True)

        # 计算advantages
        advantages = agent.compute_advantages(rewards, values, next_values, dones, gamma, gae_lambda)

        # 计算returns（也可以直接用：returns = advantages + values）
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
                 ewc_lambda=500, memory_size=1000, hidden_units=8,
                 clip_ratio=0.2, actor_lr=1e-3, critic_lr=1e-3, is_discrete=False,
                 gamma=0.99, lam=0.95):
        super().__init__(state_dim, action_dim, clip_epsilon, value_coef,
                         entropy_coef, ewc_lambda, memory_size, is_discrete)

        self.hidden_units = hidden_units
        self._tflite_models = {}
        self.actor = self._build_esp32_actor()
        self.critic = self._build_esp32_critic()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
        self.value_net = self.critic

        self.logits_layer = self.actor.layers[-1]
        self.log_std = tf.Variable(initial_value=-0.5, trainable=True, dtype=tf.float32)
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

        self.log_std = tf.Variable(initial_value=-0.5 * tf.ones(action_dim), trainable=True)

        self.clip_ratio = clip_ratio
        self.gamma = gamma
        self.lam = lam
        
        self.actor_opt = tf.keras.optimizers.Adam(actor_lr)
        self.critic_opt = tf.keras.optimizers.Adam(critic_lr)
        self.is_discrete = is_discrete
        print(f"ESP32代理初始化完成: {hidden_units}隱藏單元")

    def sigmoid(self, x):
        return 1.0 / (1.0 + tf.exp(-x))

    def _build_esp32_actor(self):
        return tf.keras.Sequential([
            tf.keras.layers.Dense(self.hidden_units, activation='relu', input_shape=(self.state_dim,)),
            tf.keras.layers.Dense(self.action_dim, activation='sigmoid')
        ], name='esp32_actor')

    def _build_esp32_critic(self):
        return tf.keras.Sequential([
            tf.keras.layers.Dense(self.hidden_units, activation='relu', input_shape=(self.state_dim,)),
            tf.keras.layers.Dense(1, activation='linear')
        ], name='esp32_critic')

    # ================= 收集一条 trajectory =================
    def collect_trajectory(self, env, max_steps=200):
        states, actions, rewards, log_probs, values = [], [], [], [], []

        state, _ = env.reset()
        for _ in range(max_steps):
            action, log_prob = self.get_action(state)
            next_state, reward, done, _, _ = env.step(action)

            states.append(state)
            actions.append(action)
            rewards.append(reward)
            log_probs.append(log_prob)
            values.append(self.critic(np.expand_dims(state, 0)).numpy()[0, 0])

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
            np.array(values, dtype=np.float32)

    # ================= 批量收集多条 trajectory =================
    def collect_trajectories(self, env, num_episodes=10, max_steps=200):
        all_states, all_actions, all_rewards, all_log_probs, all_values = [], [], [], [], []

        for _ in range(num_episodes):
            states, actions, rewards, log_probs, values = self.collect_trajectory(env, max_steps)
            all_states.append(states)
            all_actions.append(actions)
            all_rewards.append(rewards)
            all_log_probs.append(log_probs)
            all_values.append(values)

        return all_states, all_actions, all_rewards, all_log_probs, all_values

    # ================= GAE 优势函数计算 =================
    def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
        adv = np.zeros_like(rewards, dtype=np.float32)
        gae = 0
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + gamma * (1 - dones[t]) * values[t + 1] - values[t]
            gae = delta + gamma * lam * (1 - dones[t]) * gae
            adv[t] = gae
        returns = adv + values[:-1]
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

    # ========== train_step ==========
    def train_buffer_step(self,buf=None, use_ewc=True):
        states, actions, rewards, old_probs, next_states, dones = self.get_buffer_items(buf)
        if len(states) == 0:
            return 0.0

        # === 计算 returns 和 advantages ===
        values = self.critic(states).numpy().squeeze()
        next_values = self.critic(next_states).numpy().squeeze()
        deltas = rewards + self.gamma * next_values * (1 - dones) - values

        advantages = []
        adv = 0.0
        for delta, done in zip(deltas[::-1], dones[::-1]):
            adv = delta + self.gamma * self.lam * adv * (1 - done)
            advantages.insert(0, adv)
        advantages = np.array(advantages, dtype=np.float32)
        returns = advantages + values

        # === TensorFlow 训练 ===
        with tf.GradientTape() as tape:
            # 策略分布
            logits = self.actor(states)

            if self.is_discrete:
                dist = tfp.distributions.Categorical(logits=logits)
                log_probs = dist.log_prob(actions)
                entropy = dist.entropy()
            else:
                mean = logits
                log_std = tf.zeros_like(mean)  # 简化: 固定 std
                dist = tfp.distributions.Normal(loc=mean, scale=tf.exp(log_std))
                log_probs = tf.reduce_sum(dist.log_prob(actions), axis=1)
                entropy = tf.reduce_sum(dist.entropy(), axis=1)

            # old log probs
            old_log_probs = np.log(old_probs + 1e-8)

            # 比率
            ratio = tf.exp(log_probs - old_log_probs)

            # surrogate loss
            surr1 = ratio * advantages
            surr2 = tf.clip_by_value(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages
            actor_loss = -tf.reduce_mean(tf.minimum(surr1, surr2))

            # critic loss
            values_pred = self.critic(states)
            critic_loss = tf.reduce_mean(tf.square(returns - values_pred))

            # 总 loss
            loss = actor_loss + self.vf_coef * critic_loss - self.ent_coef * tf.reduce_mean(entropy)

        # === 更新参数 ===
        grads = tape.gradient(loss, self.actor.trainable_variables + self.critic.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.actor.trainable_variables + self.critic.trainable_variables))

        print(f"[Train] Loss={loss.numpy():.4f}, Actor={actor_loss.numpy():.4f}, Critic={critic_loss.numpy():.4f}")

        # 清空缓存
        self.reset_buffer()
        return loss.numpy()

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

    def ewc_regularization(self, current_params):
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

    def learn(self, buf,use_ewc=True, total_timesteps=1000000):
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


class PPOBuffer:
    def __init__(self, state_dim, action_dim, buffer_size=512, gamma=0.99):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.buffer_size = buffer_size
        self.gamma = gamma

        self.states = np.zeros((buffer_size, state_dim), dtype=np.float32)
        self.actions = np.zeros(buffer_size, dtype=np.int32)  # 存整数动作
        self.rewards = np.zeros(buffer_size, dtype=np.float32)
        self.values = np.zeros((buffer_size, action_dim), dtype=np.float32)
        self.next_states = np.zeros((buffer_size, state_dim), dtype=np.float32)
        self.probs = np.zeros((buffer_size, action_dim), dtype=np.float32)
        self.dones = np.zeros(buffer_size, dtype=np.float32)

        self.ptr = 0
        self.num_samples = 0
        self.pointer = 0
    def _init_buffer(self):
        """Initialize replay buffer with correct dimensions"""
        self.states = np.zeros((self.buffer_size, self.state_dim))
        self.actions = np.zeros((self.buffer_size, self.action_dim))
        self.probs = np.zeros((self.buffer_size, self.action_dim))  # ← Critical fix
        self.rewards = np.zeros(self.buffer_size)
        self.dones = np.zeros(self.buffer_size, dtype=bool)
        self.pointer = 0
    def is_full(self):
        return self.num_samples == self.buffer_size

    def finish_path(self, last_value=0):
        """计算 GAE 或 Returns，这里简单使用 returns = rewards + last_value"""
        self.returns = np.zeros_like(self.rewards)
        running_return = last_value
        for t in reversed(range(self.num_samples)):
            running_return = self.rewards[t] + self.gamma * running_return * (1 - self.dones[t])
            self.returns[t] = running_return
 
    def clear(self):
        self.ptr = 0
        self.num_samples = 0
 
    def store(self, state, action, prob, reward, next_state, done, value):
        
        if prob.shape != (self.action_dim,):
            # Handle the mismatch - choose appropriate strategy
            if len(prob) > self.action_dim:
                prob = prob[:self.action_dim]  # Truncate
            else:
                prob = np.pad(prob, (0, self.action_dim - len(prob)))  # Pad
        idx = self.ptr % self.buffer_size
        self.states[idx] = state
        self.actions[idx] = action
        self.probs[idx] = prob
        self.rewards[idx] = reward
        self.next_states[idx] = next_state  # 存储next_state
        self.dones[idx] = done
        self.values[idx] = value
        self.ptr += 1
        self.num_samples = min(self.num_samples + 1, self.buffer_size)

    
    def get_buffer_items(self):
        """返回 numpy 格式的完整batch"""
        return (
            np.array(self.states[:self.num_samples], dtype=np.float32),
            np.array(self.actions[:self.num_samples]),
            np.array(self.rewards[:self.num_samples], dtype=np.float32),
            np.array(self.probs[:self.num_samples], dtype=np.float32),
            np.array(self.next_states[:self.num_samples], dtype=np.float32),
            np.array(self.dones[:self.num_samples], dtype=np.bool_)
        )

    def get_batch(self, batch_size=64):
        """返回mini-batch生成器"""
        idxs = np.arange(self.num_samples)
        np.random.shuffle(idxs)

        for start in range(0, self.num_samples, batch_size):
            end = start + batch_size
            batch_idx = idxs[start:end]

            states_batch = self.states[batch_idx]
            actions_batch = self.actions[batch_idx]
            old_probs_batch = self.probs[batch_idx]
            returns_batch = self.returns[batch_idx]
            values_batch = self.values[batch_idx]
            next_states_batch = self.next_states[batch_idx]  # 新增
            dones_batch = self.dones[batch_idx]

            yield (states_batch, actions_batch, old_probs_batch, returns_batch, 
                   values_batch, next_states_batch, dones_batch)

class PPOAgent(ESP32OnlinePPOFisherAgent):
    def __init__(self, state_dim, action_dim, hidden_units=64, 
                 learning_rate=0.001, clip_ratio=0.2, gamma=0.99, lam=0.95):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_units = hidden_units
        self.clip_ratio = clip_ratio
        self.gamma = gamma
        self.lam = lam        
        super().__init__(state_dim, action_dim)
        self.buffer = PPOBuffer(state_dim, action_dim)
        # 初始化优化器
        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        
        # 构建网络
        self.actor = self._build_actor()
        self.critic = self._build_critic()
        
        # 初始化经验缓冲区
        self.buffer_size = 10000
        self._init_buffer() 


    def _init_buffer(self):
        """初始化经验缓冲区"""
        self.states = np.zeros((self.buffer_size, self.state_dim))
        self.actions = np.zeros(self.buffer_size, dtype=np.int32)  # 动作应该是整数
        self.probs = np.zeros((self.buffer_size, self.action_dim))
        self.rewards = np.zeros(self.buffer_size)
        self.dones = np.zeros(self.buffer_size, dtype=bool)
        self.pointer = 0
        self.buffer_full = False    

        
    def collect_and_train(self, env, num_episodes):
        for episode in range(num_episodes):
            state = env.reset()
            episode_reward = 0
            
            for step in range(1000):  # 最大步数
                # 1. 选择动作（增加数值稳定处理）
                action_probs = self.actor.predict(state.reshape(1, -1), verbose=0)[0]
                
                # 数值稳定化处理
                action_probs = np.clip(action_probs, 1e-8, 1.0)  # 防止过小或过大的值
                action_probs = action_probs / np.sum(action_probs)  # 重新归一化
                
                # 检查概率和是否为1（允许小的误差）
                prob_sum = np.sum(action_probs)
                if abs(prob_sum - 1.0) > 1e-6:
                    print(f"警告: 概率和不为1 ({prob_sum:.6f})，进行强制归一化")
                    action_probs = action_probs / prob_sum
                
                # 安全地选择动作
                try:
                    action = np.random.choice(len(action_probs), p=action_probs)
                except ValueError as e:
                    print(f"动作选择错误: {e}")
                    print(f"动作概率: {action_probs}")
                    print(f"概率和: {np.sum(action_probs)}")
                    # 使用均匀分布作为备选
                    action_probs = np.ones(len(action_probs)) / len(action_probs)
                    action = np.random.choice(len(action_probs), p=action_probs)
                
                # 2. 与环境交互
                next_state, reward, done, _ = env.step(action)
                
                # 3. 存储经验
                state_value = self.critic.predict(state.reshape(1, -1), verbose=0)[0]
                self.buffer.store(state, action, action_probs, reward, next_state, done, state_value)
                
                state = next_state
                episode_reward += reward
                
                if done or self.buffer.is_full():
                    # 处理完整轨迹
                    if done:
                        last_value = 0
                    else:
                        last_value = self.critic.predict(state.reshape(1, -1), verbose=0)[0][0]
                    
                    self.buffer.finish_path(last_value)
                    self.train_ppo()
                    self.buffer.clear()
                    
                    if done:
                        break
            
            if episode % 10 == 0:
                print(f"Episode {episode}, Reward: {episode_reward}")


    def train_ppo(self):
        """使用包含next_states的数据进行PPO训练"""
        # 获取完整数据用于分析或监控（可选）
        states, actions, rewards, probs, next_states, dones = self.buffer.get_buffer_items()
        print(f"训练数据形状: states{states.shape}, rewards{rewards.shape}")
        
        # Mini-batch训练
        for batch_idx, batch in enumerate(self.buffer.get_batch(batch_size=64)):
            states_b, actions_b, old_probs_b, returns_b, old_values_b, next_states_b, dones_b = batch
            
            # 修正：使用batch对应的rewards，而不是完整的rewards数组
            # 从完整数据中提取对应batch的rewards
            start_idx = batch_idx * 64
            end_idx = min((batch_idx + 1) * 64, len(rewards))
            rewards_b = rewards[start_idx:end_idx]
            
            # 确保维度匹配
            if len(rewards_b) != len(states_b):
                # 如果长度不匹配，使用buffer中的rewards（如果buffer存储了batch对应的rewards）
                rewards_b = self.buffer.rewards[start_idx:end_idx]
            
            # 使用next_states计算TD目标
            next_values = self.critic.predict(next_states_b)  # 注意：应该是self.critic而不是self.critic_network
            next_values = next_values.flatten()  # 确保是一维数组
            
            # 计算TD目标（修正维度）
            td_targets = rewards_b + self.buffer.gamma * next_values * (1 - dones_b.astype(np.float32))
            
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
                self.store(state, action, action_probs, reward, done, self.pointer)
                
                state = next_state
                self.pointer = (self.pointer + 1) % self.buffer_size

    def compute_advantages(self, rewards, values, dones, next_value):
        advantages = np.zeros_like(rewards)
        last_advantage = 0
        
        # 逆向计算GAE
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_values = next_value
            else:
                next_values = values[t + 1] * (1 - dones[t])
            
            delta = rewards[t] + self.gamma * next_values - values[t]
            advantages[t] = delta + self.gamma * self.lam * (1 - dones[t]) * last_advantage
            last_advantage = advantages[t]
        
        return advantages  # 形状应该是 [batch_size]
        
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















