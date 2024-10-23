import numpy as np
import random
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6, beta=0.4):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        max_priority = self.priorities.max() if self.buffer else 1.0
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.position] = (state, action, reward, next_state, done)
        self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        if len(self.buffer) == self.capacity:
            priorities = self.priorities
        else:
            priorities = self.priorities[:self.position]
        
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()
        
        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)
        samples = [self.buffer[idx] for idx in indices]
        
        weights = (len(self.buffer) * probabilities[indices]) ** (-self.beta)
        weights /= weights.max()
        
        return samples, indices, weights

    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority

class DuelingDQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DuelingDQN, self).__init__()
        self.feature = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )
        self.advantage = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_size)
        )
        self.value = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        x = self.feature(x)
        advantage = self.advantage(x)
        value = self.value(x)
        return value + advantage - advantage.mean()

class ImprovedQLearningAgent:
    def __init__(self, state_size, action_size, learning_rate=0.0005, gamma=0.99, epsilon=1.0, 
                 epsilon_decay=0.995, epsilon_min=0.01, memory_size=100000, batch_size=64):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = PrioritizedReplayBuffer(memory_size)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.total_trades = 0
        self.winning_trades = 0
        
        self.model = DuelingDQN(state_size, action_size).to(self.device)
        self.target_model = DuelingDQN(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # Remove the TensorBoard writer
        # self.writer = SummaryWriter()
        self.update_target_model()

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)
        if action != 0 and reward > 0:
            self.winning_trades += 1

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        act_values = self.model(state)
        action = np.argmax(act_values.cpu().data.numpy())
        if action != 0:  # If not holding
            self.total_trades += 1
        return action

    def replay(self, step):
        if len(self.memory.buffer) < self.batch_size:
            return
        
        samples, indices, weights = self.memory.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*samples)
        
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(np.array(actions)).to(self.device)
        rewards = torch.FloatTensor(np.array(rewards)).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(np.array(dones)).to(self.device)
        weights = torch.FloatTensor(weights).to(self.device)
        
        current_q_values = self.model(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.target_model(next_states).detach().max(1)[0]
        expected_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        losses = F.mse_loss(current_q_values.squeeze(), expected_q_values, reduction='none')
        loss = (losses * weights).mean()
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update priorities
        new_priorities = losses.detach().cpu().numpy()
        self.memory.update_priorities(indices, new_priorities)
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        return loss.item()

    def load(self, name):
        self.model.load_state_dict(torch.load(name))

    def save(self, name):
        torch.save(self.model.state_dict(), name)

    def get_action_distribution(self):
        total = sum(self.action_counts.values())
        distribution = {action: count / total for action, count in self.action_counts.items()}
        logger.info(f"Action distribution: {distribution}")
        return distribution

    def log_q_values(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.model(state)
        logger.debug(f"Q-values for current state: {q_values.cpu().numpy()}")

    def log_model_summary(self):
        logger.info("Model Summary:")
        logger.info(self.model)

    def log_memory_stats(self):
        if len(self.memory) > 0:
            rewards = [experience[2] for experience in self.memory]
            logger.info(f"Memory stats - Size: {len(self.memory)}, "
                        f"Max reward: {max(rewards)}, Min reward: {min(rewards)}, "
                        f"Mean reward: {sum(rewards) / len(rewards)}")

    def calculate_metrics(self):
        metrics = super().calculate_metrics()
        metrics.update({
            'total_trades': self.total_trades,
            'win_rate': self.winning_trades / self.total_trades if self.total_trades > 0 else 0,
        })
        return metrics

    # ... (implement other necessary methods like calculate_sharpe_ratio, calculate_max_drawdown, etc.)

