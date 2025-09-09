import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np
import json
import tensorflow_probability as tfd

import datetime
from collections import deque
import os
import tensorflow_model_optimization as tfmot
import zlib
import base64
import gym
from gym import spaces

from typing import Union
from typing import Dict
from typing import *


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



    def train_stepV(self, states, actions, advantages, old_probs, returns, use_ewc=True):
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
            x = x[None, ...].astype(np.float32)
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
        print(f"✅ Fisher matrix & optimal params saved to {path}")

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
        return tfd.Bernoulli(probs=logits)

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

 
class TensorFlowESP32BaseExporter:
    def __init__(self, policy_model=None):
        self.model = policy_model
        self.converter = None

    def create_representative_dataset(self, env, num_samples=1000):
        """创建代表性数据集"""
        representative_data = []
        for _ in range(num_samples):
            obs = env.reset()
            if isinstance(obs, tuple):
                obs = obs[0]
            representative_data.append(obs.astype(np.float32))
        return np.array(representative_data)

    def convert_to_tflite(self, representative_data=None, quantize=True, prune=True):
        """转换为TFLite模型"""
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)

        if quantize:
            # 设置量化参数
            converter.optimizations = [tf.lite.Optimize.DEFAULT]

            if representative_data is not None:
                def representative_dataset():
                    for data in representative_data:
                        yield [data.reshape(1, -1)]

                converter.representative_dataset = representative_dataset
                converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
                converter.inference_input_type = tf.int8
                converter.inference_output_type = tf.int8

        if prune:
            # 启用剪枝（如果适用）
            pass

        tflite_model = converter.convert()
        return tflite_model

    def save_ota_base_package(self, output_path, representative_data=None,
                         firmware_version="1.0.0", prune=True, quantize=True):
        """创建OTA包"""
        # 转换模型
        tflite_model = self.convert_to_tflite(representative_data, quantize, prune)

        # 保存TFLite模型
        tflite_path = output_path.replace('.json', '.tflite')
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)
        print(f"Saved TFLite model to {tflite_path}")

        # 创建OTA元数据
        ota_metadata = {
            "firmware_version": firmware_version,
            "model_format": "tflite",
            "input_shape": list(self.model.input_shape[1:]),
            "output_shapes": [output.shape[1:].as_list() for output in self.model.outputs],  # Convert each to list
            "quantized": quantize,
            "pruned": prune,
            "file_size": len(tflite_model),
            "checksum": self._calculate_checksum(tflite_model)
        }



        # 保存OTA包
        with open(output_path, 'w') as f:
            json.dump(ota_metadata, f, indent=2)

        print(f"Saved OTA package to {output_path}")
        print(f"OTA Metadata: {json.dumps(ota_metadata, indent=2)}")

    def _calculate_checksum(self, data):
        """计算简单的校验和"""
        return hash(data) % 1000000


# -----------------------------
# 獨立的 TensorFlow → ESP32 匯出器（不再繼承 Agent）
# -----------------------------
class TensorFlowESP32Exporter:
    def __init__(self, model_or_path: Union[str, tf.keras.Model]):
        """
        Args:
            model_or_path: 已訓練 Keras 模型或可由 tf.keras.models.load_model 載入的路徑
                           **注意**: 不支援僅權重檔（*.weights.h5)。請改用 .keras 或 SavedModel
        """
        if isinstance(model_or_path, str):
            # FIX: 只允許可 load 的完整模型
            self.model = tf.keras.models.load_model(model_or_path)
        else:
            self.model = model_or_path

        self.quantized_model_bytes: Optional[bytes] = None
        self.fisher_matrix: Optional[Dict[str, np.ndarray]] = None
        self.optimal_params: Optional[Dict[str, np.ndarray]] = None

    # 量化
    def apply_quantization(self, representative_data: np.ndarray) -> bytes:
        """
        將模型量化為 int8,適合 ESP32。
        """
        print("Applying post-training quantization...")

        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

        # 定義 generator 函數
        def representative_dataset_gen():
            for i in range(min(100, len(representative_data))):
                # 每次返回一個 batch (1, input_shape)
                yield [representative_data[i:i + 1].astype(np.float32)]

        # 注意這裡要傳函數，不要傳 generator 對象
        converter.representative_dataset = representative_dataset_gen

        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8

        tflite_model = converter.convert()

        # 保存到文件示例
        with open('quantized_model.tflite', 'wb') as f:
            f.write(tflite_model)

        print("Quantization completed! Model saved as quantized_model.tflite")
        return tflite_model


    def _representative_dataset_gen(self, dataset: np.ndarray):
        # 注意：這裡是函數，yield 產生數據
        for i in range(min(100, len(dataset))):
            yield [dataset[i:i + 1].astype(np.float32)]


    def prune_model(self, target_sparsity: float = 0.5) -> tf.keras.Model:
        print(f"Pruning model to {target_sparsity * 100}% sparsity...")

        # 確保模型已編譯
        if not hasattr(self.model, 'loss') or self.model.loss is None:
            self.model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                loss='mse',
                metrics=['mae']
            )

        # 剪枝代碼
        pruning_params = {
            'pruning_schedule': tfmot.sparsity.keras.ConstantSparsity(
                target_sparsity, begin_step=0, frequency=100
            )
        }

        pruned_model = tfmot.sparsity.keras.prune_low_magnitude(
            self.model, **pruning_params
        )

        pruned_model.compile(
            optimizer=self.model.optimizer,
            loss=self.model.loss,
            metrics=self.model.metrics
        )

        # 微調可以在這裡進行
        # pruned_model.fit(x_train, y_train, epochs=2, validation_data=(x_val, y_val))

        self.model = tfmot.sparsity.keras.strip_pruning(pruned_model)
        print("Pruning completed!")
        return self.model

    # 剪枝

    # Fisher（對分類輸出安全處理）
    def compute_fisher_matrix(self, dataset: np.ndarray, num_samples: int = 1000) -> Dict[str, np.ndarray]:
        print("Computing Fisher Information Matrix...")
        vars_list = self.model.trainable_variables
        self.optimal_params = {v.name: v.numpy().copy() for v in vars_list}
        fisher = {v.name: np.zeros_like(v.numpy()) for v in vars_list}

        @tf.function
        def grads_on_input(x):
            with tf.GradientTape() as tape:
                y = self.model(x, training=False)
                # 若輸出不是機率，轉 logits 為機率（避免 log(0)）
                if y.dtype != tf.float32:
                    y = tf.cast(y, tf.float32)
                # 小偏移避免 log(0)
                logp = tf.math.log(tf.nn.softmax(y) + 1e-8)
                # 對 batch 取均值 → 純量
                obj = tf.reduce_mean(tf.reduce_sum(logp, axis=-1))
            return tape.gradient(obj, vars_list)

        n = min(num_samples, len(dataset))
        for i in range(n):
            if i % 100 == 0:
                print(f"Processing sample {i}/{n}")
            x = tf.convert_to_tensor(dataset[i:i+1], dtype=tf.float32)
            grads = grads_on_input(x)
            for v, g in zip(vars_list, grads):
                if g is not None:
                    fisher[v.name] += (g.numpy() ** 2) / float(n)

        self.fisher_matrix = fisher
        print("Fisher matrix computation completed!")
        return fisher

    # 壓縮
    def compress_for_esp32(self, tflite_bytes: bytes) -> Dict[str, Any]:
        print("Compressing model for ESP32...")
        comp = zlib.compress(tflite_bytes)
        original_size = len(tflite_bytes)
        compressed_size = len(comp)
        compression_ratio = original_size / compressed_size if compressed_size > 0 else float('inf')
        print(f"Compression ratio: {compression_ratio:.2f}x")
        print(f"Original: {original_size} bytes, Compressed: {compressed_size} bytes")
        return {
            'compressed_model': comp,
            'original_size': original_size,
            'compressed_size': compressed_size,
            'compression_ratio': compression_ratio,  # FIX: 供 metadata 使用
        }

    # OTA 組包
    def create_ota_package(self,
                           representative_data: np.ndarray,
                           firmware_version: str = "1.0.0",
                           prune: bool = True,
                           quantize: bool = True) -> Dict[str, Any]:
        print("Creating OTA package...")
        if prune:
            self.prune_model(target_sparsity=0.5)

        self.compute_fisher_matrix(representative_data)

        if quantize:
            tflite_bytes = self.apply_quantization(representative_data)
        else:
            converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
            tflite_bytes = converter.convert()

        comp = self.compress_for_esp32(tflite_bytes)
        simplified_fisher = self._simplify_fisher_matrix()
        metadata = {
            'firmware_version': firmware_version,
            'model_type': 'TFLite',
            'input_shape': list(self.model.input_shape[1:]) if self.model.input_shape else None,
            'output_shape': list(self.model.output_shape[1:]) if self.model.output_shape else None,
            'compression_ratio': comp['compression_ratio'],
            'timestamp': datetime.now().isoformat(),
            'quantization': quantize,
            'pruning': prune,
        }

        # FIX: JSON 不能存 bytes，先 Base64
        model_data_b64 = base64.b64encode(comp['compressed_model']).decode('ascii')

        ota_package = {
            'metadata': metadata,
            'model_data_b64': model_data_b64,
            'fisher_matrix': simplified_fisher,
            'optimal_params': self._prepare_optimal_params(),
            'crc32': self._calculate_crc(comp['compressed_model']),
        }
        return ota_package

    def _simplify_fisher_matrix(self) -> Dict[str, Any]:
        simplified = {}
        assert self.fisher_matrix is not None
        for name, fm in self.fisher_matrix.items():
            thr = np.percentile(fm, 80.0)
            mask = fm > thr
            values = fm[mask]
            indices = np.where(mask)
            simplified[name] = {
                'values': values.astype(np.float32).tolist(),
                'indices': [idx.astype(np.int32).tolist() for idx in indices],
                'shape': list(fm.shape),
                'threshold': float(thr),
            }
        return simplified

    def _prepare_optimal_params(self) -> Dict[str, Any]:
        assert self.optimal_params is not None
        return {name: arr.astype(np.float32).tolist() for name, arr in self.optimal_params.items()}

    def _calculate_crc(self, data: bytes) -> int:
        return zlib.crc32(data)

    def save_ota_package(self, output_path: str, representative_data: np.ndarray, **kwargs) -> None:
        ota = self.create_ota_package(representative_data, **kwargs)

        # JSON 檔（Base64 模型）
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(ota, f, indent=2, ensure_ascii=False)

        # 二進位（含 bytes）— 使用 pickle
        binary_path = output_path.replace('.json', '.bin')
        import pickle
        with open(binary_path, 'wb') as f:
            pickle.dump(ota, f)

        print(f"OTA package saved to {output_path} and {binary_path}")
        print(f"JSON size: {os.path.getsize(output_path)} bytes")
        print(f"Binary size: {os.path.getsize(binary_path)} bytes")

class PlantHVACEnv(gym.Env):
    """
    自定義 Plant HVAC 環境
    狀態 (state_dim=5):
        [溫度, 濕度, 光照, CO2濃度, 土壤濕度]
    動作 (action_dim=4):
        0 = 空調降溫
        1 = 加濕器增濕
        2 = 開燈補光
        3 = 通風降CO2

    獎勵:
        根據與最佳生長區間的差距給分
    """
    def __init__(self):
        super(PlantHVACEnv, self).__init__()

        # 觀測空間: 連續狀態 (5維)
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0, 200.0, 0.0], dtype=np.float32),
            high=np.array([50.0, 100.0, 2000.0, 2000.0, 100.0], dtype=np.float32),
            dtype=np.float32
        )

        # 動作空間: 4個離散動作
        self.action_space = spaces.Discrete(4)

        # 初始狀態
        self.state = None
        self.reset()

        # 最佳生長區間 (目標區域)
        self.optimal_ranges = {
            "temp": (22, 28),      # 溫度 (°C)
            "humidity": (60, 75),  # 濕度 (%)
            "light": (400, 800),   # 光照 (lux)
            "co2": (350, 600),     # CO2 ppm
            "soil": (40, 60),      # 土壤濕度 (%)
        }

    def reset(self):
        """重置環境"""
        self.state = self.observation_space.sample()
        return self.state

    def step(self, action):
        """
        根據 action 更新狀態
        """
        temp, humidity, light, co2, soil = self.state

        # 簡單模擬 HVAC 控制
        if action == 0:   # 降溫
            temp -= 1.5
        elif action == 1: # 加濕
            humidity += 2.0
        elif action == 2: # 補光
            light += 50
        elif action == 3: # 通風
            co2 -= 30

        # 加入隨機擾動 (模擬外部環境影響)
        temp += np.random.uniform(-0.5, 0.5)
        humidity += np.random.uniform(-1, 1)
        light += np.random.uniform(-20, 20)
        co2 += np.random.uniform(-10, 10)
        soil += np.random.uniform(-1, 1)

        # 更新狀態
        self.state = np.array([temp, humidity, light, co2, soil], dtype=np.float32)

        # 計算 reward
        reward = self._calculate_reward()

        # 終止條件：偏離太嚴重
        done = self._is_done()

        return self.state, reward, done, {}

    def _calculate_reward(self):
        """根據與最佳生長區間的差距計算 reward"""
        temp, humidity, light, co2, soil = self.state
        score = 0

        def range_penalty(value, optimal_range):
            low, high = optimal_range
            if value < low:
                return -(low - value)
            elif value > high:
                return -(value - high)
            else:
                return +1.0  # 在區間內加分

        score += range_penalty(temp, self.optimal_ranges["temp"])
        score += range_penalty(humidity, self.optimal_ranges["humidity"])
        score += range_penalty(light, self.optimal_ranges["light"])
        score += range_penalty(co2, self.optimal_ranges["co2"])
        score += range_penalty(soil, self.optimal_ranges["soil"])

        return score

    def _is_done(self):
        """當狀態超出極限值，結束 episode"""
        temp, humidity, light, co2, soil = self.state
        if temp < 0 or temp > 50:
            return True
        if humidity < 0 or humidity > 100:
            return True
        if light < 0 or light > 2000:
            return True
        if co2 < 100 or co2 > 3000:
            return True
        if soil < 0 or soil > 100:
            return True
        return False

class PlantLLLHVACEnv:
    def __init__(self, seq_len=20, n_features=5, temp_init=25.0, humid_init=0.5,
                 latent_dim=64, mode="growing"):
        self.seq_len = seq_len
        self.temp_init = temp_init
        self.humid_init = humid_init
        self.n_features = n_features  # 現在有5個特徵: temp, humid, health, light, co2
        self.mode = mode  # "growing", "flowering", "seeding"

        # 構建encoder
        self.encoder = self._build_encoder(seq_len, n_features, latent_dim)

        # LLL模型
        self.lll_model = self._build_lll_model(latent_dim, hidden_dim=64, output_dim=3)
        self.fisher_matrix = None
        self.prev_weights = None
        self.memory = deque(maxlen=1000)  # 簡單的記憶緩衝區

        # 不同模式的理想環境參數（添加光照和CO2範圍）
        self.mode_params = {
            "growing": {
                "temp_range": (22, 28),
                "humid_range": (0.4, 0.7),
                "vpd_range": (0.8, 1.5),
                "light_range": (300, 600),  # lux
                "co2_range": (400, 800)  # ppm
            },
            "flowering": {
                "temp_range": (20, 26),
                "humid_range": (0.4, 0.6),
                "vpd_range": (1.0, 1.8),
                "light_range": (500, 800),  # lux
                "co2_range": (600, 1000)  # ppm
            },
            "seeding": {
                "temp_range": (24, 30),
                "humid_range": (0.5, 0.7),
                "vpd_range": (0.7, 1.3),
                "light_range": (200, 400),  # lux
                "co2_range": (400, 600)  # ppm
            }
        }

        # 初始化環境變量
        self.light = 500.0  # 初始光照 (lux)
        self.co2 = 600.0  # 初始CO2濃度 (ppm)

        # 初始化狀態變量
        self.reset()

    def _build_encoder(self, seq_len, n_features, latent_dim):
        """構建序列編碼器"""
        return tf.keras.Sequential([
            tf.keras.layers.LSTM(64, input_shape=(seq_len, n_features), return_sequences=True),
            tf.keras.layers.LSTM(32),
            tf.keras.layers.Dense(latent_dim, activation='relu')
        ])

    def _build_lll_model(self, input_dim, hidden_dim, output_dim):
        """構建終身學習模型"""
        return tf.keras.Sequential([
            tf.keras.layers.Dense(hidden_dim, activation='relu', input_shape=(input_dim,)),
            tf.keras.layers.Dense(hidden_dim, activation='relu'),
            tf.keras.layers.Dense(output_dim, activation='softmax')
        ])

    def update_lll_model(self, sequence_input, true_label=None):
        """
        更新LLL模型並返回預測結果

        Args:
            sequence_input: 輸入序列數據
            true_label: 真實標籤（可選，用於訓練）

        Returns:
            soft_label: 軟標籤預測概率
        """
        # 編碼序列數據
        latent_representation = self.encoder(sequence_input)

        # 獲取預測
        soft_label = self.lll_model(latent_representation)

        # 如果有真實標籤，則進行訓練
        if true_label is not None:
            self._train_lll_model(latent_representation, true_label)

        return soft_label.numpy()[0]  # 返回第一個batch的預測

    def _train_lll_model(self, latent_input, true_label):
        """
        訓練LLL模型
        """
        # 計算EWC正則化損失
        ewc_loss = self._compute_ewc_loss()

        with tf.GradientTape() as tape:
            predictions = self.lll_model(latent_input, training=True)

            # 計算交叉熵損失
            ce_loss = tf.keras.losses.sparse_categorical_crossentropy(
                true_label, predictions
            )

            # 總損失 = 交叉熵損失 + EWC正則化
            total_loss = tf.reduce_mean(ce_loss) + ewc_loss

        # 更新模型權重
        gradients = tape.gradient(total_loss, self.lll_model.trainable_variables)
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        optimizer.apply_gradients(zip(gradients, self.lll_model.trainable_variables))

    def _compute_ewc_loss(self):
        """計算EWC正則化損失"""
        if self.fisher_matrix is None or self.prev_weights is None:
            return 0.0

        ewc_loss = 0.0
        ewc_lambda = 500  # EWC正則化強度

        for i, (current_var, prev_var, fisher) in enumerate(
                zip(self.lll_model.trainable_variables,
                    self.prev_weights,
                    self.fisher_matrix)):
            ewc_loss += tf.reduce_sum(
                ewc_lambda * 0.5 * fisher * tf.square(current_var - prev_var)
            )

        return ewc_loss

    def save_model_knowledge(self):
        """保存當前模型的知識（用於EWC）"""
        # 保存當前權重
        self.prev_weights = [tf.identity(var) for var in self.lll_model.trainable_variables]

        # 計算Fisher信息矩陣（這裡簡化處理）
        self.fisher_matrix = [
            tf.ones_like(var) * 0.1 for var in self.lll_model.trainable_variables
        ]

    def reset(self):
        """重置環境狀態"""
        self.temp = self.temp_init
        self.humid = self.humid_init
        self.light = 500.0
        self.co2 = 600.0
        self.health = 2  # 初始狀態設為"無法判定"
        self.t = 0
        self.prev_action = np.zeros(4, dtype=int)

        # 初始化序列數據
        self.current_sequence = np.zeros((1, self.seq_len, self.n_features))

        # 填充初始序列
        for i in range(self.seq_len):
            self.current_sequence[0, i] = [self.temp, self.humid, self.health, self.light, self.co2]

        return self._get_state()

    def _get_state(self):
        """獲取當前狀態"""
        return np.array([self.health, self.temp, self.humid, self.light, self.co2], dtype=np.float32)

    def update_sequence(self, new_data_point):
        """
        更新數據序列，添加新的數據點

        Args:
            new_data_point: 新的傳感器數據點 [temp, humid, health, light, co2]
        """
        # 將序列向前移動一位，移除最舊的數據
        self.current_sequence = np.roll(self.current_sequence, shift=-1, axis=1)

        # 在序列末尾添加新的數據點
        self.current_sequence[0, -1] = new_data_point

    def _calculate_enhanced_vpd(self, temp, humid, light, co2):
        """
        計算增強的VPD，考慮光照和CO2的影響
        """
        # 基礎VPD計算
        base_vpd = self.calc_vpd(temp, humid)

        # 光照對VPD的影響因子
        light_factor = np.clip((light - 200) / 600, 0.8, 1.2)

        # CO2對VPD的影響因子
        co2_factor = np.clip(1.0 - (co2 - 400) / 1000, 0.8, 1.0)

        # 增強的VPD計算
        enhanced_vpd = base_vpd * light_factor * co2_factor

        return enhanced_vpd

    def calc_vpd(self, temp, humid):
        """計算VPD（蒸汽壓差）"""
        # 飽和蒸汽壓計算（Tetens公式）
        es = 0.6108 * np.exp(17.27 * temp / (temp + 237.3))
        # 實際蒸汽壓
        ea = es * humid
        # VPD
        vpd = es - ea
        return max(vpd, 0.1)

    def step(self, action, params=None, true_label=None):
        """
        執行動作並返回新的狀態、獎勵等信息
        """
        if params is None:
            params = {}

        ac, humi, heat, dehumi = action

        # 環境動力學
        self.temp += (-0.5 if ac == 1 else 0.2) + (0.5 if heat == 1 else 0.0)
        self.humid += (0.05 if humi == 1 else -0.02) + (-0.03 if heat == 1 else 0.0) + (-0.05 if dehumi == 1 else 0.0)

        # 光照和CO2的自然變化
        self.light += np.random.normal(0, 20)
        self.co2 += np.random.normal(0, 10)

        # 邊界限制
        self.temp = np.clip(self.temp, 15, 35)
        self.humid = np.clip(self.humid, 0, 1)
        self.light = np.clip(self.light, 100, 1000)
        self.co2 = np.clip(self.co2, 300, 1200)

        # 根據當前模式獲取理想環境參數
        mode_param = self.mode_params[self.mode]
        temp_min, temp_max = mode_param["temp_range"]
        humid_min, humid_max = mode_param["humid_range"]
        vpd_min, vpd_max = mode_param["vpd_range"]
        light_min, light_max = mode_param["light_range"]
        co2_min, co2_max = mode_param["co2_range"]

        # 計算增強的VPD
        vpd_current = self._calculate_enhanced_vpd(self.temp, self.humid, self.light, self.co2)

        # 健康判定
        temp_ok = temp_min <= self.temp <= temp_max
        humid_ok = humid_min <= self.humid <= humid_max
        vpd_ok = vpd_min <= vpd_current <= vpd_max
        light_ok = light_min <= self.light <= light_max
        co2_ok = co2_min <= self.co2 <= co2_max

        # 綜合健康判定
        optimal_conditions = sum([temp_ok, humid_ok, vpd_ok, light_ok, co2_ok])

        if optimal_conditions >= 4:
            self.health = 0  # 健康
        elif optimal_conditions >= 2:
            self.health = 1  # 亞健康
        else:
            self.health = 2  # 不健康

        # 更新序列數據
        new_data_point = np.array([self.temp, self.humid, self.health, self.light, self.co2])
        self.update_sequence(new_data_point)

        # LLL模型預測
        seq_input_tf = tf.convert_to_tensor(self.current_sequence, dtype=tf.float32)

        if true_label is not None and not isinstance(true_label, tf.Tensor):
            true_label_tf = tf.convert_to_tensor(true_label, dtype=tf.int32)
        else:
            true_label_tf = true_label

        # 使用update_lll_model方法獲取軟標籤
        soft_label = self.update_lll_model(seq_input_tf, true_label_tf)
        flower_prob = soft_label[2]

        # 計算獎勵
        health_reward = {0: 2.0, 1: 0.5, 2: -1.0}[self.health]
        energy_cost = params.get("energy_penalty", 0.1) * np.sum(action)
        switch_penalty = params.get("switch_penalty_per_toggle", 0.2) * np.sum(np.abs(action - self.prev_action))

        # 環境因子獎勵
        vpd_ideal = (vpd_min + vpd_max) / 2
        vpd_reward = -abs(vpd_current - vpd_ideal) * params.get("vpd_penalty", 2.0)

        light_ideal = (light_min + light_max) / 2
        light_reward = -abs(self.light - light_ideal) * params.get("light_penalty", 0.5)

        co2_ideal = (co2_min + co2_max) / 2
        co2_reward = -abs(self.co2 - co2_ideal) * params.get("co2_penalty", 0.3)

        learning_reward = 0
        if true_label is not None:
            pred_class = np.argmax(soft_label)
            true_class = true_label if isinstance(true_label, (int, np.integer)) else true_label.numpy()
            learning_reward = 0.5 if pred_class == true_class else -0.3

        # 軟標籤獎勵
        soft_label_bonus = 0
        if self.mode == "flowering":
            soft_label_bonus = flower_prob * params.get("flower_bonus", 0.5)
        elif self.mode == "seeding":
            soft_label_bonus = soft_label[1] * params.get("seed_bonus", 0.5)
        else:
            soft_label_bonus = soft_label[0] * params.get("grow_bonus", 0.5)

        reward = (health_reward - energy_cost - switch_penalty +
                  vpd_reward + light_reward + co2_reward +
                  learning_reward + soft_label_bonus)

        self.prev_action = action
        self.t += 1
        done = self.t >= self.seq_len

        info = {
            "latent_soft_label": soft_label,
            "flower_prob": flower_prob,
            "temp": self.temp,
            "humid": self.humid,
            "vpd": vpd_current,
            "light": self.light,
            "co2": self.co2,
            "learning_reward": learning_reward,
            "soft_label_bonus": soft_label_bonus,
            "health_status": self.health,
            "health_status_text": ["健康", "亞健康", "不健康"][self.health],
            "optimal_conditions": optimal_conditions
        }

        return self._get_state(), reward, done, info



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


# 使用示例
def train_ESP32PPOAgent():
    # 创建环境和智能体
    env = PlantHVACEnv(mode="flowering")

    #agent       = ESP32PPOAgent(state_dim=5, action_dim=4)
    agent = ESP32PPOAgent(state_dim=5, action_dim=4, hidden_units=8)
    # 顯示模型信息
    print("模型參數數量:")
    print(f"Actor: {agent._count_params(agent.actor)}")
    print(f"Critic: {agent._count_params(agent.critic)}")
    # 收集经验
    experiences = collect_experiences(agent, env, num_episodes=10)

    # 分析收集到的经验
    print(f"总共收集了 {len(experiences)} 条经验")

    # 计算统计信息
    rewards = [exp['reward'] for exp in experiences]
    health_statuses = [exp['info']['health_status'] for exp in experiences]

    print(f"平均奖励: {np.mean(rewards):.3f}")
    print(f"健康比例: {np.mean([1 if s == 0 else 0 for s in health_statuses]) * 100:.1f}%")
    print(f"不健康比例: {np.mean([1 if s == 1 else 0 for s in health_statuses]) * 100:.1f}%")


    # 導出ESP32所需文件
    agent.export_for_esp32()

    # 測試推理
    test_state = np.array([0, 25.0, 0.5, 500.0, 600.0], dtype=np.float32)
    action = agent.get_action_esp32(test_state)
    value = agent.get_value_esp32(test_state)

    print(f"測試狀態: {test_state}")
    print(f"預測動作: {action}")
    print(f"預測價值: {value}")
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

