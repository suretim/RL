import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np
import tensorflow_probability as tfp

import datetime
from collections import deque
import os

from util_env import PlantLLLHVACEnv,PlantHVACEnv


# state_dim = 5  # 健康狀態、溫度、濕度、光照、CO2
# action_dim = 4  # 4個控制動作
class LifelongPPOBaseAgent:
    def __init__(self, state_dim=5, action_dim=4,
                 clip_epsilon=0.2, value_coef=0.5, entropy_coef=0.01,
                 ewc_lambda=500, memory_size=1000):
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.ewc_lambda = ewc_lambda  # EWC正则化强度
        self.env=PlantHVACEnv()
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

    def _compute_ewc_loss(self):
        """ EWC(Elastic Weight Consolidation)Regulation Loss"""
        ewc_loss = 0
        if not self.fisher_matrices:
            return ewc_loss

        # 对每个之前学习的任务计算EWC损失
        for task_id in self.fisher_matrices:
            fisher_matrix = self.fisher_matrices[task_id]
            optimal_params = self.optimal_params[task_id]

            current_params = self.actor.trainable_variables + self.critic.trainable_variables

            for i, (current_param, optimal_param, fisher) in enumerate(
                    zip(current_params, optimal_params, fisher_matrix)):
                # EWC损失: λ/2 * Σ F_i * (θ_i - θ*_i)^2
                ewc_loss += tf.reduce_sum(
                    self.ewc_lambda * 0.5 * fisher * tf.square(current_param - optimal_param)
                )

        return ewc_loss

    def _compute_fisher_matrix(self, experiences):
        """计算Fisher信息矩阵来衡量参数的重要性"""
        states, actions, _, old_probs, _ = experiences

        fisher_matrix = []
        with tf.GradientTape(persistent=True) as tape:
            # 计算当前策略的概率
            new_probs = self.actor(states, training=True)
            actions_float = tf.cast(actions, tf.float32)

            # 计算对数概率
            new_action_probs = tf.reduce_sum(
                new_probs * actions_float + (1 - new_probs) * (1 - actions_float), axis=1
            )
            log_probs = tf.math.log(new_action_probs + 1e-8)

        # 计算每个参数的Fisher信息
        trainable_vars = self.actor.trainable_variables + self.critic.trainable_variables
        for var in trainable_vars:
            gradients = tape.gradient(log_probs, var)
            if gradients is not None:
                # Fisher信息 ≈ 梯度的平方
                fisher_info = tf.reduce_mean(tf.square(gradients))
                fisher_matrix.append(fisher_info)
            else:
                fisher_matrix.append(tf.constant(0.0, dtype=tf.float32))

        return fisher_matrix

    def save_task_knowledge(self, task_experiences):
        """保存当前任务的知识"""
        # 保存最优参数
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
        """回放之前任务的经验来防止遗忘"""
        if len(self.task_memory) == 0:
            return 0

        # 随机采样之前任务的经验
        indices = np.random.choice(len(self.task_memory), min(batch_size, len(self.task_memory)), replace=False)
        batch = [self.task_memory[i] for i in indices]

        # 组织成批量数据
        states = tf.stack([exp[0] for exp in batch])
        actions = tf.stack([exp[1] for exp in batch])
        advantages = tf.convert_to_tensor([exp[2] for exp in batch], dtype=tf.float32)
        old_probs = tf.stack([exp[3] for exp in batch])
        returns = tf.convert_to_tensor([exp[4] for exp in batch], dtype=tf.float32)

        # 进行训练（但不计算EWC损失，因为这是记忆回放）
        with tf.GradientTape() as tape:
            new_probs = self.actor(states, training=True)
            new_values = self.critic(states, training=True)

            # 计算PPO损失（与原始train_step相同）
            actions_float = tf.cast(actions, tf.float32)
            old_action_probs = tf.reduce_sum(old_probs * actions_float + (1 - old_probs) * (1 - actions_float), axis=1)
            new_action_probs = tf.reduce_sum(new_probs * actions_float + (1 - new_probs) * (1 - actions_float), axis=1)

            ratio = new_action_probs / (old_action_probs + 1e-8)
            clipped_ratio = tf.clip_by_value(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)

            policy_loss = -tf.reduce_mean(tf.minimum(ratio * advantages, clipped_ratio * advantages))
            value_loss = tf.reduce_mean(tf.square(returns - tf.squeeze(new_values)))

            entropy = -tf.reduce_mean(
                new_probs * tf.math.log(new_probs + 1e-8) +
                (1 - new_probs) * tf.math.log(1 - new_probs + 1e-8)
            )

            total_loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

        # 应用梯度
        grads = tape.gradient(total_loss, self.actor.trainable_variables + self.critic.trainable_variables)
        self.optimizer.apply_gradients(
            zip(grads, self.actor.trainable_variables + self.critic.trainable_variables)
        )

        return total_loss

    def train_step(self, states, actions, advantages, old_probs, returns, use_ewc=True):
        """
        增强的训练步骤，兼容 one-hot actions 和单输出 critic
        states: [batch_size, state_dim]
        actions: [batch_size, action_dim] one-hot
        advantages: [batch_size] 或 [batch_size, action_dim]
        old_probs: [batch_size, action_dim]
        returns: [batch_size]
        """
        with tf.GradientTape() as tape:
            # 预测新动作概率和状态值
            new_probs = self.actor(states, training=True)  # [batch_size, action_dim]
            new_values = tf.squeeze(self.critic(states, training=True), axis=-1)  # [batch_size]

            # 计算选中动作的概率
            # actions 是 one-hot，[batch_size, action_dim]
            selected_new_probs = tf.reduce_sum(new_probs * actions, axis=1)  # [batch_size]
            selected_old_probs = tf.reduce_sum(old_probs * actions, axis=1)  # [batch_size]

            # 计算 PPO ratio
            ratio = selected_new_probs / (selected_old_probs + 1e-8)
            clipped_ratio = tf.clip_by_value(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)

            # 如果 advantages 是 [batch_size, action_dim]，取对应动作
            if len(advantages.shape) == 2:
                advantages = tf.reduce_sum(advantages * actions, axis=1)  # [batch_size]

            # PPO policy loss
            policy_loss = -tf.reduce_mean(tf.minimum(ratio * advantages, clipped_ratio * advantages))

            # value loss
            value_loss = tf.reduce_mean(tf.square(returns - new_values))

            # entropy bonus
            entropy = -tf.reduce_mean(
                new_probs * tf.math.log(new_probs + 1e-8) +
                (1 - new_probs) * tf.math.log(1 - new_probs + 1e-8)
            )

            # 基础损失
            base_loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

            # EWC正则化
            ewc_loss = self._compute_ewc_loss() if use_ewc and self.fisher_matrices else 0

            # 总损失
            total_loss = base_loss + ewc_loss

        # 梯度更新
        grads = tape.gradient(total_loss, self.actor.trainable_variables + self.critic.trainable_variables)
        self.optimizer.apply_gradients(
            zip(grads, self.actor.trainable_variables + self.critic.trainable_variables)
        )

        return total_loss, policy_loss, value_loss, entropy, ewc_loss

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



    def xtrain_stepV(self, states, actions, advantages, old_probs, returns, use_ewc=True):
        """增强的训练步骤，包含持续学习机制"""
        with tf.GradientTape() as tape:
            # 原始PPO损失
            new_probs = self.actor(states, training=True)
            new_values = self.critic(states, training=True)

            actions_float = tf.cast(actions, tf.float32)
            old_action_probs = tf.reduce_sum(old_probs * actions_float + (1 - old_probs) * (1 - actions_float), axis=1)
            new_action_probs = tf.reduce_sum(new_probs * actions_float + (1 - new_probs) * (1 - actions_float), axis=1)

            ratio = new_action_probs / (old_action_probs + 1e-8)
            clipped_ratio = tf.clip_by_value(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)

            policy_loss = -tf.reduce_mean(tf.minimum(ratio * advantages, clipped_ratio * advantages))
            value_loss = tf.reduce_mean(tf.square(returns - tf.squeeze(new_values)))

            entropy = -tf.reduce_mean(
                new_probs * tf.math.log(new_probs + 1e-8) +
                (1 - new_probs) * tf.math.log(1 - new_probs + 1e-8)
            )

            # 基础损失
            base_loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

            # EWC正则化损失（防止遗忘）
            ewc_loss = self._compute_ewc_loss() if use_ewc and self.fisher_matrices else 0

            # 总损失 = 基础损失 + EWC正则化
            total_loss = base_loss + ewc_loss

        # 应用梯度
        grads = tape.gradient(total_loss, self.actor.trainable_variables + self.critic.trainable_variables)
        self.optimizer.apply_gradients(
            zip(grads, self.actor.trainable_variables + self.critic.trainable_variables)
        )

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


class LifelongESP32PPOAgent(LifelongPPOBaseAgent):
    """專為ESP32設計的輕量級PPO代理,繼承自LifelongPPOAgent"""

    def __init__(self, state_dim=5, action_dim=4,
                 clip_epsilon=0.2, value_coef=0.5, entropy_coef=0.01,
                 ewc_lambda=500, memory_size=1000, hidden_units=8):
        """
        初始化ESP32專用代理

        Args:
            hidden_units: 隱藏層神經元數量,根據ESP32內存調整
        """
        super().__init__(state_dim, action_dim, clip_epsilon, value_coef,
                         entropy_coef, ewc_lambda, memory_size)

        self.hidden_units = hidden_units
        self._tflite_models = {}

        # 重新構建更小的網絡
        self.actor = self._build_esp32_actor()
        self.critic = self._build_esp32_critic()

        # 使用更小的學習率
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

        print(f"ESP32代理初始化完成: {hidden_units}隱藏單元")

    def _build_esp32_actor(self):
        """構建適合ESP32的輕量級Actor網絡"""
        return tf.keras.Sequential([
            tf.keras.layers.Dense(self.hidden_units, activation='relu',
                                  input_shape=(self.state_dim,),
                                  kernel_regularizer=tf.keras.regularizers.l2(0.001)),
            tf.keras.layers.Dense(self.action_dim, activation='sigmoid',
                                  kernel_regularizer=tf.keras.regularizers.l2(0.001))
        ], name='esp32_actor')

    def _build_esp32_critic(self):
        """構建適合ESP32的輕量級Critic網絡"""
        return tf.keras.Sequential([
            tf.keras.layers.Dense(self.hidden_units, activation='relu',
                                  input_shape=(self.state_dim,),
                                  kernel_regularizer=tf.keras.regularizers.l2(0.001)),
            tf.keras.layers.Dense(1, activation='linear',
                                  kernel_regularizer=tf.keras.regularizers.l2(0.001))
        ], name='esp32_critic')

    def save_tflite_model(self, filepath, model_type='actor'):
        """保存TFLite模型到文件"""
        if model_type not in self._tflite_models:
            self.convert_to_tflite(model_type)

        # 確保目錄存在
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)

        with open(filepath, 'wb') as f:
            f.write(self._tflite_models[model_type])
        print(f"TFLite {model_type} model saved to {filepath}")

    def save_policy_model_savedmodel(self, model, model_name, model_type='actor'):
        """
        保存 Keras 模型为 SavedModel (.pb) 格式到 save_models 目录
        """
        # 目录结构： save_models/<model_name>_<model_type>/
        save_dir = "saved_models"
        os.makedirs(save_dir, exist_ok=True)

        model_dir = os.path.join(save_dir, f"{model_name}_{model_type}")
        model.save(model_dir, save_format='tf')  # 保存为 SavedModel 格式

        print(f"Policy model saved to {model_dir}")

    def save_policy_model(self, model_name, model_type='actor'):
        """保存TFLite模型到 saved_models 目录"""

        if model_type not in self._tflite_models:
            self.convert_to_tflite(model_type)

        # 确保 saved_models 目录存在
        save_dir = "saved_models"
        os.makedirs(save_dir, exist_ok=True)

        # 文件名规则：<model_name>_<model_type>.tflite
        filepath = os.path.join(save_dir, f"{model_name}_{model_type}.tflite")

        with open(filepath, 'wb') as f:
            f.write(self._tflite_models[model_type])

        print(f"TFLite {model_type} model saved to {filepath}")

    def load_tflite_model(self, filepath, model_type='actor'):
        """從文件加載TFLite模型"""
        with open(filepath, 'rb') as f:
            tflite_model = f.read()

        self._tflite_models[model_type] = tflite_model
        return tflite_model

    def convert_to_tflite(self, model_type='actor', quantize=True, optimize_size=True):
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

    def export_for_esp32(self, base_path="../esp32_model"):
        """導出ESP32所需的所有文件"""
        os.makedirs(base_path, exist_ok=True)

        # 轉換並保存TFLite模型
        self.convert_to_tflite('actor')
        self.convert_to_tflite('critic')

        # 使用正確的方法名
        self.save_tflite_model(f"{base_path}/actor.tflite", 'actor')
        self.save_tflite_model(f"{base_path}/critic.tflite", 'critic')

        # 生成C頭文件
        self._generate_c_header(f"{base_path}/actor.tflite", f"{base_path}/actor.h", 'actor_model')
        self._generate_c_header(f"{base_path}/critic.tflite", f"{base_path}/critic.h", 'critic_model')

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


class ESP32PPOWithFisherAgent(LifelongESP32PPOAgent):
    def __init__(self, state_dim=5, action_dim=4, hidden_units=8, ewc_lambda=500):
        super().__init__(state_dim, action_dim, hidden_units)
        self.ewc_lambda = ewc_lambda
        self.fisher_matrix = None
        self.optimal_params = None

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

class OnlineESP32PPOAgent(LifelongESP32PPOAgent):
    """專為ESP32設計的輕量級PPO代理，支持 online EWC 和 TFLite 導出"""

    def __init__(self, state_dim=5, action_dim=4,
                 clip_epsilon=0.2, value_coef=0.5, entropy_coef=0.01,
                 ewc_lambda=500, memory_size=1000, hidden_units=8):
        super().__init__(state_dim, action_dim, clip_epsilon, value_coef,
                         entropy_coef, ewc_lambda, memory_size)

        self.hidden_units = hidden_units
        self._tflite_models = {}

        self.actor = self._build_esp32_actor()
        self.critic = self._build_esp32_critic()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

        # EWC online
        self.online_fisher = None
        self.optimal_params = None
        self.ema_decay = 0.99
        self.fisher_update_frequency = 1
        self.update_counter = 0

        print(f"ESP32代理初始化完成: {hidden_units}隱藏單元")

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

    # ------------------ 新增 policy 方法 ------------------
    def policy(self, obs):
        """
        返回動作分佈 (Bernoulli) 供 PPO loss 計算
        obs: tf.Tensor, shape (batch_size, state_dim)
        """
        logits = self.actor(obs)
        return tfp.Bernoulli(probs=logits)

    # ------------------ EWC online 更新 ------------------
    def update_online_fisher(self, obs: tf.Tensor, action: tf.Tensor):
        self.update_counter += 1
        if self.update_counter % self.fisher_update_frequency != 0:
            return

        if self.online_fisher is None:
            self._initialize_online_fisher()

        with tf.GradientTape() as tape:
            dist = self.policy(obs)
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
            self.update_online_fisher(obs_batch[i:i+1], action_batch[i:i+1])

        return {
            'total_loss': total_loss.numpy(),
            'policy_loss': policy_loss.numpy(),
            'value_loss': value_loss.numpy(),
            'entropy_loss': entropy_loss.numpy(),
            'ewc_loss': ewc_loss.numpy()
        }


class LifelongPPOAgent(LifelongPPOBaseAgent):
    def __init__(self, state_dim=5, action_dim=4):
        super().__init__(state_dim=state_dim, action_dim=action_dim)
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.policy = self._build_policy_network()



    def train_step(self, states, actions, advantages, old_probs, returns, use_ewc=True):
        """Training step using PPO"""
        with tf.GradientTape() as tape:
            # Get the new policy outputs
            mean, log_std = self.policy(states)
            std = tf.exp(log_std)
            dist = tfp.distributions.Normal(mean, std)

            # Calculate log probabilities of taken actions
            new_probs = dist.log_prob(actions)
            old_probs = tf.convert_to_tensor(old_probs, dtype=tf.float32)

            # Calculate the ratio (for clipping)
            ratio = tf.exp(new_probs - old_probs)

            # PPO objective (with clipping)
            clip_ratio = 0.2
            clipped_ratio = tf.clip_by_value(ratio, 1 - clip_ratio, 1 + clip_ratio)
            policy_loss = -tf.reduce_mean(tf.minimum(ratio * advantages, clipped_ratio * advantages))

            # Value loss (if applicable, you can add a value network)
            # value_loss = tf.reduce_mean((returns - value_preds)**2)

            # Total loss (if you include value loss and entropy)
            entropy_bonus = tf.reduce_mean(dist.entropy())
            total_loss = policy_loss - 0.01 * entropy_bonus  # Regularization via entropy

            # Apply gradients
            gradients = tape.gradient(total_loss, self.policy.trainable_variables)
            optimizer = tf.keras.optimizers.Adam(learning_rate=3e-4)
            optimizer.apply_gradients(zip(gradients, self.policy.trainable_variables))

        return total_loss

    def _build_policy_network(self):
        """Build the policy network"""
        inputs = tf.keras.layers.Input(shape=(self.state_dim,))
        x = tf.keras.layers.Dense(64, activation='relu')(inputs)
        x = tf.keras.layers.Dense(64, activation='relu')(x)

        # Action distribution parameters - using Dense layers
        mean = tf.keras.layers.Dense(self.action_dim)(x)
        log_std = tf.keras.layers.Dense(self.action_dim)(x)

        return tf.keras.Model(inputs=inputs, outputs=[mean, log_std])

    def sample_action(self, state):
        """Sample action based on current policy"""
        mean, log_std = self.policy(state)
        std = tf.exp(log_std)  # Convert log_std to actual standard deviation
        dist = tfp.distributions.Normal(mean, std)  # Define the Gaussian distribution
        action = dist.sample()  # Sample an action from the distribution
        return action.numpy()  # Convert to numpy array for usage in environment

    def get_action(self, state):
        """Get action for environment interaction"""
        return self.sample_action(state)

    def learn(self, states, actions, advantages, old_probs, returns, use_ewc=True, total_timesteps=1000000):
        """Simulated training process"""
        print(f"Training for {total_timesteps} timesteps...")

        for i in range(total_timesteps // 1000):
            self.train_step(states, actions, advantages, old_probs, returns, use_ewc=True)
            if i % 100 == 0:
                print(f"Step {i * 1000}/{total_timesteps}")



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
    advantages = compute_advantages(rewards, values, next_values, dones, gamma, gae_lambda)

    # 计算returns（也可以直接用：returns = advantages + values）
    returns = compute_returns(rewards, dones, gamma)

    return states, actions, advantages, old_probs, returns

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
            action, old_prob = agent.get_action(state, return_probs=True)

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
        self.probs = np.zeros((buffer_size, action_dim), dtype=np.float32)
        self.dones = np.zeros(buffer_size, dtype=np.float32)

        self.ptr = 0
        self.num_samples = 0

    def store(self, state, action, prob, reward, done, value):
        """存储经验"""
        idx = self.ptr % self.buffer_size  # 循环覆盖
        self.states[idx] = state
        self.actions[idx] = action
        self.probs[idx] = prob
        self.rewards[idx] = reward
        self.dones[idx] = done
        self.values[idx] = value
        self.ptr += 1
        self.num_samples = min(self.num_samples + 1, self.buffer_size)

    def is_full(self):
        return self.num_samples == self.buffer_size

    def finish_path(self, last_value=0):
        """计算 GAE 或 Returns，这里简单使用 returns = rewards + last_value"""
        self.returns = np.zeros_like(self.rewards)
        running_return = last_value
        for t in reversed(range(self.num_samples)):
            running_return = self.rewards[t] + self.gamma * running_return * (1 - self.dones[t])
            self.returns[t] = running_return

    def get_batch(self, batch_size=64):
        """返回 batch，actions 自动返回整数，训练时做 one-hot"""
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

            yield states_batch, actions_batch, old_probs_batch, returns_batch, values_batch

    def clear(self):
        self.ptr = 0
        self.num_samples = 0


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
#if __name__ == "__main__":
#    physical_devices = tf.config.list_physical_devices('GPU')
#    if physical_devices:
#        tf.config.experimental.set_memory_growth(physical_devices[0], True)
#    train_LifelongPPOBaseAgent()



    # 使用最简单的版本避免tf.function问题
    # train_ppo_simple()
    # 配置GPU

    # 使用简化版本的智能体
    # train_ppo_with_lll()
    #
    #    model.save("demo_model.keras")

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

