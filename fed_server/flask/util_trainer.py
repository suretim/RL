import tensorflow as tf
import numpy as np
from util_agent import ESP32PPOAgent
from util_env import PlantLLLHVACEnv

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


class LLLTrainer():
    def __init__(self, lll_model=None,  learning_rate=0.001, ewc_lambda=0.4):
        #super().__init__(ewc_lambda)

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.ewc_lambda = ewc_lambda  # EWC 正则化强度
        self.fisher_info = None       # 存储重要参数的 Fisher 信息
        self.old_fisher_matrix = None
        self.old_params = None        # 存储旧任务参数
        if lll_model is not None:
            self.lll_model = lll_model
        else:
            self.lll_model = PlantLLLHVACEnv().lll_model
        self.latent_dim = self.lll_model.input_shape[1]
    def compute_fisher(self, data, labels):
        """
        估算旧任务参数的重要性 (Fisher 信息)
        """
        fisher = []
        for var in self.lll_model.trainable_variables:
            fisher.append(tf.zeros_like(var))
        # 简单示例，只计算单批次梯度平方
        with tf.GradientTape() as tape:
            predictions = self.lll_model(data, training=False)
            loss = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(labels, predictions))
        grads = tape.gradient(loss, self.lll_model.trainable_variables)
        fisher = [g**2 for g in grads]  # Fisher 信息近似为梯度平方
        return fisher

    def update_ewc(self, data, labels):
        """
        保存旧任务参数和 Fisher 信息
        """
        self.old_params = [var.numpy() for var in self.lll_model.trainable_variables]
        self.fisher_info = self.compute_fisher(data, labels)

    def _compute_ewc_loss(self):
        """
        计算 EWC 正则化损失
        """
        if self.old_params is None or self.fisher_info is None:
            return 0.0
        ewc_loss = 0.0
        for old, fisher, var in zip(self.old_params, self.fisher_info, self.lll_model.trainable_variables):
            ewc_loss += tf.reduce_sum(fisher * (var - old)**2)
        return self.ewc_lambda * ewc_loss

    def _train_lll_model(self, latent_inputs, true_labels):
        """
        訓練LLL模型（支持批量）
        """
        ewc_loss = self._compute_ewc_loss()

        with tf.GradientTape() as tape:
            predictions = self.lll_model(latent_inputs, training=True)
            ce_loss = tf.keras.losses.sparse_categorical_crossentropy(true_labels, predictions)
            ce_loss = tf.reduce_mean(ce_loss)
            total_loss = ce_loss + ewc_loss

        gradients = tape.gradient(total_loss, self.lll_model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.lll_model.trainable_variables))
        return total_loss





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


# --------------------------
# 示例：多任务训练
# --------------------------
#latent_dim = 64
#num_classes = 3
#batch_size = 32
def test(latent_dim, num_classes,batch_size):
    num_epochs_per_task = 3
    num_tasks = 3
    env = PlantLLLHVACEnv(seq_len=10,   mode="growing")
    lll_model=env.lll_model
    # 创建简单模型
    '''
    lll_model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(latent_dim,)),
        tf.keras.layers.Dense(latent_dim, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    '''
    trainer = LLLTrainer(lll_model, learning_rate=0.001, ewc_lambda=0.4)

    # 模拟多任务数据
    for task_id in range(num_tasks):
        print(f"\n=== Training Task {task_id+1} ===")
        num_samples = 200
        latent_features = np.random.randn(num_samples, latent_dim).astype(np.float32)
        labels = np.random.randint(0, num_classes, size=(num_samples,)).astype(np.int32)

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
                print(f"  Epoch {epoch+1}, Last batch loss: {loss:.4f}")

        # 训练完当前任务后，更新 EWC 信息
        trainer.update_ewc(latent_features, labels)
