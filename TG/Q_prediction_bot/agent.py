import numpy as np
import random
from collections import deque
import tensorflow as tf
from tensorflow.keras.regularizers import l2
from tensorflow import keras

print(f"TensorFlow version: {tf.__version__}")
print(f"Keras version: {tf.keras.__version__}")

class ImprovedQLearningAgent:
    """
    An improved Q-learning agent using a deep neural network (Deep Q-Network).
    """

    def __init__(self, state_size, action_size, learning_rate=0.001, gamma=0.95, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01, memory_size=2000, batch_size=32):
        """
        Initialize the Deep Q-Network agent.

        Args:
        state_size (int): Dimension of the state space
        action_size (int): Number of possible actions
        learning_rate (float): Learning rate for the optimizer
        gamma (float): Discount factor for future rewards
        epsilon (float): Initial exploration rate
        epsilon_decay (float): Decay rate for epsilon
        epsilon_min (float): Minimum value for epsilon
        memory_size (int): Size of the replay memory
        batch_size (int): Number of samples to use for training
        """
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=memory_size)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.delta = 0.01  # For delta-epsilon method

        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()

    def _build_model(self):
        """
        Build the neural network model for the DQN.

        Returns:
        keras.Model: Compiled Keras model
        """
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(32, input_dim=self.state_size, activation='relu', kernel_regularizer=l2(0.01)),
            tf.keras.layers.Dense(32, activation='relu', kernel_regularizer=l2(0.01)),
            tf.keras.layers.Dense(self.action_size, activation='linear', kernel_regularizer=l2(0.01))
        ])
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate))
        return model

    def update_target_model(self):
        """Update the target network with weights from the main network."""
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        """
        Store a transition in the replay memory.

        Args:
        state: Current state
        action: Action taken
        reward: Reward received
        next_state: Next state
        done: Whether the episode is done
        """
        self.memory.append((state, action, reward, next_state, done))

    def get_action(self, state):
        """
        Choose an action using an epsilon-greedy policy.

        Args:
        state: Current state

        Returns:
        int: Chosen action
        """
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        q_values = self.model.predict(state, verbose=0)
        return np.argmax(q_values[0])

    def replay(self):
        """Train the network on a batch of transitions from the replay memory."""
        if len(self.memory) < self.batch_size:
            return

        minibatch = random.sample(self.memory, self.batch_size)
        states = np.array([i[0] for i in minibatch])
        actions = np.array([i[1] for i in minibatch])
        rewards = np.array([i[2] for i in minibatch])
        next_states = np.array([i[3] for i in minibatch])
        dones = np.array([i[4] for i in minibatch])

        states = np.squeeze(states)
        next_states = np.squeeze(next_states)

        targets = rewards + self.gamma * (np.amax(self.target_model.predict_on_batch(next_states), axis=1)) * (1 - dones)
        targets_full = self.model.predict_on_batch(states)

        ind = np.array([i for i in range(self.batch_size)])
        targets_full[[ind], [actions]] = targets

        self.model.fit(states, targets_full, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def delta_epsilon_update(self, reward):
        """
        Update epsilon based on the reward received.

        Args:
        reward (float): Reward received from the last action
        """
        if reward > 0:
            self.epsilon = max(self.epsilon_min, self.epsilon - self.delta)
        else:
            self.epsilon = min(1.0, self.epsilon + self.delta)

    def load(self, name):
        """Load model weights from a file."""
        self.model.load_weights(name)

    def save(self, name):
        """Save model weights to a file."""
        self.model.save_weights(name)
