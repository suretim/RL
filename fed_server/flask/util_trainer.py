import tensorflow as tf
import numpy as np
from util_hvac_agent import ESP32PPOAgent
from util_env import PlantLLLHVACEnv

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

# --------------------------
# 示例：多任务训练
# --------------------------
latent_dim = 64
num_classes = 3
batch_size = 32
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
