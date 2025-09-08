import tensorflow as tf
import numpy as np
from util_hvac_PPO import LifelongPPOBaseAgent
import tensorflow_probability as tfp


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
