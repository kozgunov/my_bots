import numpy as np
import random
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim

class SimplerDQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SimplerDQN, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        return self.layers(x)

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
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SimplerDQN(state_size, action_size).to(self.device)
        self.target_model = SimplerDQN(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()

        # New attributes for tracking performance
        self.wins = 0
        self.losses = 0
        self.total_trades = 0
        self.cumulative_reward = 0
        self.win_streak = 0
        self.max_win_streak = 0

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

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
        
        # Update performance metrics
        self.total_trades += 1
        self.cumulative_reward += reward
        
        if reward > 0:
            self.wins += 1
            self.win_streak += 1
            self.max_win_streak = max(self.max_win_streak, self.win_streak)
        elif reward < 0:
            self.losses += 1
            self.win_streak = 0

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
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.model(state)
        return q_values.argmax().item()

    def replay(self, batch_size):
        """Train the network on a batch of transitions from the replay memory."""
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*minibatch)
        
        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        next_states = np.array(next_states)
        dones = np.array(dones)

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        current_q_values = self.model(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.target_model(next_states).max(1)[0].detach()
        target_q_values = rewards + (self.gamma * next_q_values * (1 - dones))

        loss = self.criterion(current_q_values, target_q_values.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return loss.item()

    def load(self, name):
        """Load model weights from a file."""
        self.model.load_state_dict(torch.load(name))
        self.model.eval()

    def save(self, name):
        """Save model weights to a file."""
        torch.save(self.model.state_dict(), name)

    def get_performance_metrics(self):
        win_rate = self.wins / self.total_trades if self.total_trades > 0 else 0
        avg_reward = self.cumulative_reward / self.total_trades if self.total_trades > 0 else 0
        return {
            'win_rate': win_rate,
            'total_trades': self.total_trades,
            'wins': self.wins,
            'losses': self.losses,
            'avg_reward': avg_reward,
            'max_win_streak': self.max_win_streak,
            'current_win_streak': self.win_streak
        }

    def reset_performance_metrics(self):
        self.wins = 0
        self.losses = 0
        self.total_trades = 0
        self.cumulative_reward = 0
        self.win_streak = 0
        self.max_win_streak = 0
