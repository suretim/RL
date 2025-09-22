import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np
from collections import deque
import random

#state_dim = 5  # 健康狀態、溫度、濕度、光照、CO2
#action_dim = 4  # 4個控制動作
class LifelongPPOAgent:
    def __init__(self, state_dim=5, action_dim=4,
                 clip_epsilon=0.2, value_coef=0.5, entropy_coef=0.01,
                 ewc_lambda=500, memory_size=1000):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.ewc_lambda = ewc_lambda  # EWC正则化强度

        # 构建网络
        self.actor = self._build_actor()
        self.critic = self._build_critic()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=3e-4)

        # 持续学习相关
        self.task_memory = deque(maxlen=memory_size)  # 存储之前任务的经验
        self.fisher_matrices = {}  # 存储每个任务的Fisher信息矩阵
        self.optimal_params = {}  # 存储每个任务的最优参数
        self.current_task_id = 0
    def _count_params(self, model):
        """計算模型參數數量"""
        return sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])

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

    def get_action(self, state):
        """Get action for environment interaction (same as select_action)"""
        action, probs = self.select_action(state)
        return action, probs

    def _compute_ewc_loss(self):
        """计算EWC（Elastic Weight Consolidation）正则化损失"""
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


import tensorflow as tf
import numpy as np
from collections import deque
import os


class ESP32PPOAgent(LifelongPPOAgent):
    """專為ESP32設計的輕量級PPO代理，繼承自LifelongPPOAgent"""

    def __init__(self, state_dim=5, action_dim=4,
                 clip_epsilon=0.2, value_coef=0.5, entropy_coef=0.01,
                 ewc_lambda=500, memory_size=1000, hidden_units=8):
        """
        初始化ESP32專用代理

        Args:
            hidden_units: 隱藏層神經元數量，根據ESP32內存調整
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

    def load_tflite_model(self, filepath, model_type='actor'):
        """從文件加載TFLite模型"""
        with open(filepath, 'rb') as f:
            tflite_model = f.read()

        self._tflite_models[model_type] = tflite_model
        return tflite_model

    def convert_to_tflite(self, model_type='actor', quantize=True, optimize_size=True):
        """
        將模型轉換為TFLite格式，針對ESP32優化
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



class PlantHVACEnv:
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
# 辅助函数（需要确保这些函数已定义）
def calc_vpd(temp, humid):
    """计算VPD（蒸汽压差）"""
    # 饱和蒸汽压计算（Tetens公式）
    es = 0.6108 * np.exp(17.27 * temp / (temp + 237.3))
    # 实际蒸汽压
    ea = es * humid
    # VPD
    vpd = es - ea
    return max(vpd, 0.1)  # 确保VPD不为负




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
    agent = SimplePPOAgent(state_dim=3, action_dim=4)
    buffer = PPOBuffer(state_dim=3, action_dim=4, buffer_size=1024)

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


# 最简单的版本：不使用tf.function
class SimpleNoTFFunctionPPOAgent:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.actor = self._build_actor()
        self.critic = self._build_critic()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.0003)

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
        state_tensor = tf.convert_to_tensor([state], dtype=tf.float32)
        action_probs = self.actor(state_tensor, training=False)
        actions = [1 if np.random.random() < prob else 0 for prob in action_probs[0]]
        return np.array(actions), action_probs.numpy()[0]

    def get_value(self, state):
        state_tensor = tf.convert_to_tensor([state], dtype=tf.float32)
        return self.critic(state_tensor, training=False).numpy()[0, 0]

    def train_step(self, states, actions, advantages, old_probs, returns):
        """不使用tf.function的简单版本"""
        with tf.GradientTape() as tape:
            new_probs = self.actor(states, training=True)
            new_values = self.critic(states, training=True)

            # 计算损失
            actions_float = tf.cast(actions, tf.float32)
            old_action_probs = tf.reduce_sum(old_probs * actions_float + (1 - old_probs) * (1 - actions_float), axis=1)
            new_action_probs = tf.reduce_sum(new_probs * actions_float + (1 - new_probs) * (1 - actions_float), axis=1)

            ratio = new_action_probs / (old_action_probs + 1e-8)
            clipped_ratio = tf.clip_by_value(ratio, 0.8, 1.2)

            policy_loss = -tf.reduce_mean(tf.minimum(ratio * advantages, clipped_ratio * advantages))
            value_loss = tf.reduce_mean(tf.square(returns - tf.squeeze(new_values)))
            entropy = -tf.reduce_mean(new_probs * tf.math.log(new_probs + 1e-8))

            total_loss = policy_loss + 0.5 * value_loss - 0.01 * entropy

        grads = tape.gradient(total_loss, self.actor.trainable_variables + self.critic.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.actor.trainable_variables + self.critic.trainable_variables))

        return total_loss


class PPOAgent:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.actor, self.critic = self._build_networks()
        self._compile_models()  # 编译模型
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.0003)
        self.clip_epsilon = 0.2
        self.value_coef = 0.5
        self.entropy_coef = 0.01

    def _build_networks(self):
        # Actor网络 - 输出每个动作的概率
        actor_inputs = tf.keras.Input(shape=(self.state_dim,))
        x = layers.Dense(64, activation='relu')(actor_inputs)
        x = layers.Dense(64, activation='relu')(x)
        actor_outputs = layers.Dense(self.action_dim, activation='sigmoid')(x)

        actor = tf.keras.Model(actor_inputs, actor_outputs)

        # Critic网络 - 输出状态价值
        critic_inputs = tf.keras.Input(shape=(self.state_dim,))
        x = layers.Dense(64, activation='relu')(critic_inputs)
        x = layers.Dense(64, activation='relu')(x)
        critic_outputs = layers.Dense(1, activation='linear')(x)

        critic = tf.keras.Model(critic_inputs, critic_outputs)

        return actor, critic

    def _compile_models(self):
        """编译模型"""
        # 编译actor（虽然我们使用自定义训练，但编译可以避免错误）
        self.actor.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='mse'  # 占位符损失函数
        )

        # 编译critic
        self.critic.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='mse'
        )

    def select_action(self, state, training=True):
        """选择动作 - 基于概率采样"""
        state_tensor = tf.convert_to_tensor([state], dtype=tf.float32)
        action_probs = self.actor(state_tensor, training=False)

        # 对每个动作独立采样
        actions = []
        for i in range(self.action_dim):
            prob = action_probs[0, i].numpy()
            action = 1 if np.random.random() < prob else 0
            actions.append(action)

        return np.array(actions), action_probs.numpy()[0]

    @tf.function
    def train_step(self, states, actions, advantages, old_probs, returns):
        """PPO训练步骤"""
        with tf.GradientTape() as tape:
            # 计算新概率
            new_probs = self.actor(states, training=True)
            new_values = self.critic(states, training=True)

            # 计算概率比
            actions_float = tf.cast(actions, tf.float32)
            old_action_probs = tf.reduce_sum(old_probs * actions_float + (1 - old_probs) * (1 - actions_float), axis=1)
            new_action_probs = tf.reduce_sum(new_probs * actions_float + (1 - new_probs) * (1 - actions_float), axis=1)

            ratio = new_action_probs / (old_action_probs + 1e-8)

            # 裁剪的概率比
            clipped_ratio = tf.clip_by_value(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)

            # PPO损失
            policy_loss = -tf.reduce_mean(
                tf.minimum(ratio * advantages, clipped_ratio * advantages)
            )

            # 价值损失
            value_loss = tf.reduce_mean(tf.square(returns - tf.squeeze(new_values)))

            # 熵奖励（鼓励探索）
            entropy = -tf.reduce_mean(
                new_probs * tf.math.log(new_probs + 1e-8) +
                (1 - new_probs) * tf.math.log(1 - new_probs + 1e-8)
            )

            # 总损失
            total_loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

        # 计算梯度并更新
        grads = tape.gradient(
            total_loss,
            self.actor.trainable_variables + self.critic.trainable_variables
        )
        self.optimizer.apply_gradients(
            zip(grads, self.actor.trainable_variables + self.critic.trainable_variables)
        )

        return total_loss, policy_loss, value_loss, entropy


# 或者使用更简单的方法：创建一个虚拟训练方法
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

    @tf.function
    def train_step(self, states, actions, advantages, old_probs, returns):
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


def process_experiences(agent,experiences, gamma=0.99, gae_lambda=0.95):
    """
    处理经验并计算advantages和returns
    """
    # 提取数据
    states = [exp[0] for exp in experiences]
    actions = [exp[1] for exp in experiences]
    rewards = [exp[2] for exp in experiences]
    next_states = [exp[3] for exp in experiences]
    dones = [exp[4] for exp in experiences]
    old_probs = [exp[5] for exp in experiences]

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


def xcollect_experiences(agent, env, num_episodes=100, max_steps_per_episode=None):
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
            action = agent.get_action(state)

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
def trainbytask_lifelong_ppo():
    # 创建环境和智能体
    env = PlantHVACEnv(mode="flowering")

    agent = LifelongPPOAgent(state_dim=5, action_dim=4)



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

    return experiences



def train_lifelong_ppo():
    # 创建多个不同的环境（代表不同任务）
    tasks = [
        PlantHVACEnv(mode="flowering"),
        PlantHVACEnv(mode="seeding"),
        PlantHVACEnv(mode="growing"),
    ]

    agent = LifelongPPOAgent(state_dim=3, action_dim=4)

    # 按顺序学习每个任务
    for task_id, env in enumerate(tasks):
        print(f"开始学习任务 {task_id}...")

        # 训练当前任务
        experiences = collect_experiences(agent, env, num_episodes=100)
        states, actions, advantages, old_probs, returns = process_experiences(experiences)

        # 训练多个epoch
        for epoch in range(10):
            loss = agent.train_step(states, actions, advantages, old_probs, returns, use_ewc=True)

        # 保存当前任务知识
        agent.save_task_knowledge((states, actions, advantages, old_probs, returns))

        # 测试所有已学任务的性能（检查是否遗忘）
        for test_id in range(task_id + 1):
            performance = agent.test_task_performance(tasks[test_id])
            print(f"任务 {test_id} 测试性能: {performance}")

        # 回放之前任务的经验
        for _ in range(5):
            agent.replay_previous_tasks(batch_size=32)


# 修改训练循环使用SimplePPOAgent
def train_ppo_with_lll():
    # 初始化环境和智能体
    env = PlantHVACEnv()
    #agent = SimplePPOAgent(state_dim=3, action_dim=4)  # 使用简化版本
    agent=LifelongPPOAgent(state_dim=3, action_dim=4)
    buffer = PPOBuffer(state_dim=3, action_dim=4, buffer_size=2048)
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
            # 计算advantages和returns（需要你实现）
            #advantages = compute_advantages(experiences)
            #returns = compute_returns(experiences)
            states, actions, advantages, old_probs, returns=process_experiences(agent, experiences, gamma=0.99, gae_lambda=0.95)
            # 提取批量数据
            #states = tf.stack([exp[0] for exp in experiences])
            #actions = tf.stack([exp[1] for exp in experiences])
            #old_probs = tf.stack([exp[5] for exp in experiences])

            # 批量训练
            total_loss+=agent.train_step(states, actions, advantages, old_probs, returns)

            # 清空经验缓冲区
            experiences = []

        '''
        episode_reward = 0
        
        while not done:
            # 选择动作
            action, action_probs = agent.select_action(state)
            value = agent.get_value(state)

            # 执行动作
            next_state, reward, done, info = env.step(action, params)

            # 存储经验
            buffer.store(state, action, action_probs, reward, done, value)

            state = next_state
            episode_reward += reward

            # 缓冲区满时训练
            if buffer.is_full():
                last_value = agent.get_value(state) if not done else 0
                buffer.finish_path(last_value)

                # 训练PPO
                for epoch in range(10):
                    for batch in buffer.get_batch(batch_size=64):
                        loss = agent.train_step(*batch)

                print(f"Episode {episode}, PPO Loss: {loss:.4f}")
        '''
        print(f"Episode {episode}, Reward: {total_loss:.2f}")
    return agent


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
    """PPO经验回放缓冲区"""

    def __init__(self, state_dim, action_dim, buffer_size=2048, gamma=0.99, gae_lambda=0.95):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.buffer_size = buffer_size
        self.gamma = gamma
        self.gae_lambda = gae_lambda

        # 初始化所有缓冲区
        self.states = np.zeros((buffer_size, state_dim))
        self.actions = np.zeros((buffer_size, action_dim))
        self.probs = np.zeros((buffer_size, action_dim))
        self.rewards = np.zeros(buffer_size)
        self.dones = np.zeros(buffer_size)
        self.values = np.zeros(buffer_size)

        # 添加advantages和returns缓冲区
        self.advantages = np.zeros(buffer_size)
        self.returns = np.zeros(buffer_size)

        self.ptr = 0
        self.path_start_idx = 0
        self.num_samples = 0  # 添加样本计数器

    def store(self, state, action, prob, reward, done, value):
        """存储经验"""
        if self.ptr >= self.buffer_size:
            print("Buffer overflow, overwriting old experiences")
            self.ptr = 0  # 循环覆盖

        idx = self.ptr
        self.states[idx] = state
        self.actions[idx] = action
        self.probs[idx] = prob
        self.rewards[idx] = reward
        self.dones[idx] = done
        self.values[idx] = value
        self.ptr += 1
        self.num_samples = min(self.num_samples + 1, self.buffer_size)

    def finish_path(self, last_value=0):
        """完成一个episode，计算GAE和returns"""
        if self.ptr == self.path_start_idx:
            return  # 没有新数据

        path_slice = slice(self.path_start_idx, self.ptr)
        rewards = np.append(self.rewards[path_slice], last_value)
        values = np.append(self.values[path_slice], last_value)
        dones = np.append(self.dones[path_slice], 0)

        # 计算GAE (Generalized Advantage Estimation)
        deltas = rewards[:-1] + self.gamma * values[1:] * (1 - dones[:-1]) - values[:-1]

        advantages = np.zeros_like(deltas)
        advantage = 0
        for t in reversed(range(len(deltas))):
            advantage = deltas[t] + self.gamma * self.gae_lambda * advantage * (1 - dones[t])
            advantages[t] = advantage

        # 计算returns
        returns = advantages + values[:-1]

        # 标准化advantages
        if len(advantages) > 1:  # 确保有多个样本才能标准化
            advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)

        # 存储到缓冲区
        self.advantages[path_slice] = advantages
        self.returns[path_slice] = returns

        self.path_start_idx = self.ptr

    def get_batch(self, batch_size=64, require_full=False):
        """获取训练批次"""
        if require_full:
            # 只有在需要完整缓冲区时才检查
            if self.ptr < self.buffer_size:
                return None

        # 使用当前存储的样本数量
        available_samples = self.num_samples if self.num_samples > 0 else self.ptr

        if available_samples < batch_size:
            # 如果样本不足，返回所有可用样本
            batch_size = available_samples
            if batch_size == 0:
                return None

        # 随机选择索引
        indices = np.random.choice(available_samples, batch_size, replace=False)

        return (
            tf.convert_to_tensor(self.states[indices], dtype=tf.float32),
            tf.convert_to_tensor(self.actions[indices], dtype=tf.float32),
            tf.convert_to_tensor(self.advantages[indices], dtype=tf.float32),
            tf.convert_to_tensor(self.probs[indices], dtype=tf.float32),
            tf.convert_to_tensor(self.returns[indices], dtype=tf.float32)
        )

    def get_all_data(self):
        """获取所有数据（用于小批量训练）"""
        available_samples = self.num_samples if self.num_samples > 0 else self.ptr
        if available_samples == 0:
            return None

        indices = np.arange(available_samples)

        return (
            tf.convert_to_tensor(self.states[indices], dtype=tf.float32),
            tf.convert_to_tensor(self.actions[indices], dtype=tf.float32),
            tf.convert_to_tensor(self.advantages[indices], dtype=tf.float32),
            tf.convert_to_tensor(self.probs[indices], dtype=tf.float32),
            tf.convert_to_tensor(self.returns[indices], dtype=tf.float32)
        )

    def is_full(self):
        """检查缓冲区是否已满"""
        return self.ptr >= self.buffer_size

    def has_enough_samples(self, min_samples=64):
        """检查是否有足够的样本进行训练"""
        available_samples = self.num_samples if self.num_samples > 0 else self.ptr
        return available_samples >= min_samples

    def clear(self):
        """清空缓冲区"""
        self.ptr, self.path_start_idx = 0, 0
        self.num_samples = 0
        self.states.fill(0)
        self.actions.fill(0)
        self.probs.fill(0)
        self.rewards.fill(0)
        self.dones.fill(0)
        self.values.fill(0)
        self.advantages.fill(0)
        self.returns.fill(0)



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

    esp32_agent = ESP32PPOAgent(state_dim=5, action_dim=4, hidden_units=8)

    # 顯示模型信息
    print("模型參數數量:")
    print(f"Actor: {esp32_agent._count_params(esp32_agent.actor)}")
    print(f"Critic: {esp32_agent._count_params(esp32_agent.critic)}")

    # 導出ESP32所需文件
    esp32_agent.export_for_esp32()

    # 測試推理
    test_state = np.array([0, 25.0, 0.5, 500.0, 600.0], dtype=np.float32)
    action = esp32_agent.get_action_esp32(test_state)
    value = esp32_agent.get_value_esp32(test_state)

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

