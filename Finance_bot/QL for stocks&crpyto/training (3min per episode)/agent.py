import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
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

class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.fc4 = nn.Linear(64, 32)
        self.bn4 = nn.BatchNorm1d(32)
        self.fc5 = nn.Linear(32, action_size)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = F.leaky_relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = F.leaky_relu(self.bn3(self.fc3(x)))
        x = self.dropout(x)
        x = F.leaky_relu(self.bn4(self.fc4(x)))
        x = self.fc5(x)
        return x

class ImprovedQLearningAgent:
    def __init__(self, state_size, action_size, learning_rate=0.0001, gamma=0.99, 
                 epsilon_start=1.0, epsilon_end=0.01, epsilon_decay_steps=10000, 
                 memory_size=10000, batch_size=64, feature_names=None):
        logger.debug("Initializing ImprovedQLearningAgent")
        self.state_size = state_size
        self.action_size = action_size
        self.memory = PrioritizedReplayBuffer(capacity=memory_size)
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_rate = (epsilon_start - epsilon_end) / epsilon_decay_steps
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.feature_names = feature_names or []
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DQN(state_size, action_size).to(self.device)
        self.target_model = DQN(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=1000, gamma=0.1)
        self.update_target_model()
        self.action_counts = {0: 0, 1: 0, 2: 0, 3: 0}
        self.total_trades = 0
        self.winning_trades = 0
        self.gradient_norms = []

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)
        if action != 0 and reward > 0:
            self.winning_trades += 1
        self.total_trades += 1

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            action = random.randrange(self.action_size)
        else:
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            self.model.eval()
            with torch.no_grad():
                act_values = self.model(state)
            self.model.train()
            action = np.argmax(act_values.cpu().data.numpy())
        
        # Update action counts
        self.action_counts[action] += 1
        
        return action

    def replay(self, batch_size):
        if len(self.memory.buffer) < batch_size:
            return None

        samples, indices, weights = self.memory.sample(batch_size)
        states, actions, rewards, next_states, dones = zip(*samples)
        
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(np.array(actions)).to(self.device)
        rewards = torch.FloatTensor(np.array(rewards)).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(np.array(dones)).to(self.device)
        weights = torch.FloatTensor(weights).to(self.device)
        
        state_action_values = self.model(states).gather(1, actions.unsqueeze(1))
        next_state_values = self.target_model(next_states).max(1)[0].detach()
        expected_state_action_values = (next_state_values * self.gamma) * (1 - dones) + rewards
        loss = F.mse_loss(state_action_values, expected_state_action_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        
        # Calculate gradient norm
        total_norm = 0
        for p in self.model.parameters():
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        self.gradient_norms.append(total_norm)
        
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        self.scheduler.step()

        td_errors = torch.abs(state_action_values - expected_state_action_values.unsqueeze(1)).detach().cpu().numpy()
        new_priorities = td_errors + 1e-6
        self.memory.update_priorities(indices, new_priorities)

        return loss.item()

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_end, self.epsilon - self.epsilon_decay_rate)

    def calculate_metrics(self):
        return {
            'total_trades': self.total_trades,
            'win_rate': self.winning_trades / max(self.total_trades, 1),
        }

    def calculate_loss(self, state, action, reward, next_state, done):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        next_state = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
        action = torch.LongTensor([action]).to(self.device)
        reward = torch.FloatTensor([reward]).to(self.device)
        done = torch.FloatTensor([done]).to(self.device)

        state_action_value = self.model(state).gather(1, action.unsqueeze(1))
        next_state_value = self.target_model(next_state).max(1)[0].detach()
        expected_state_action_value = (next_state_value * self.gamma) * (1 - done) + reward

        loss = F.mse_loss(state_action_value, expected_state_action_value.unsqueeze(1))
        return loss.item()

    def save(self, path):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
        }, path)
        logger.info(f"Model saved to {path}")

    def load(self, path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.total_trades = checkpoint['total_trades']
        self.winning_trades = checkpoint['winning_trades']
        logger.info(f"Model loaded from {path}")

    def get_action_distribution(self):
        total = sum(self.action_counts.values())
        if total == 0:
            return {action: 0 for action in self.action_counts}
        return {action: count / total for action, count in self.action_counts.items()}

    def calculate_sharpe_ratio(self, returns, risk_free_rate=0.02):
        try:
            returns = np.array(returns)
            excess_returns = returns - risk_free_rate
            return np.mean(excess_returns) / (np.std(excess_returns) + 1e-10)
        except Exception as e:
            logger.error(f"Error calculating Sharpe Ratio: {e}")
            return None

    def calculate_max_drawdown(self, portfolio_values):
        try:
            portfolio_values = np.array(portfolio_values)
            peak = np.maximum.accumulate(portfolio_values)
            drawdown = (peak - portfolio_values) / peak
            return np.max(drawdown)
        except Exception as e:
            logger.error(f"Error calculating Max Drawdown: {e}")
            return None

    def calculate_profit_factor(self, profits):
        try:
            profits = np.array(profits)
            gains = profits[profits > 0].sum()
            losses = abs(profits[profits < 0].sum())
            return gains / (losses + 1e-10)
        except Exception as e:
            logger.error(f"Error calculating Profit Factor: {e}")
            return None

    def calculate_sortino_ratio(self, returns, risk_free_rate=0.02, target_return=0):
        try:
            returns = np.array(returns)
            excess_returns = returns - risk_free_rate
            downside_returns = excess_returns[excess_returns < target_return]
            downside_deviation = np.sqrt(np.mean(downside_returns**2))
            return (np.mean(excess_returns) - target_return) / (downside_deviation + 1e-10)
        except Exception as e:
            logger.error(f"Error calculating Sortino Ratio: {e}")
            return None

    def calculate_calmar_ratio(self, returns, max_drawdown):
        try:
            annual_return = np.mean(returns) * 252  # Assuming 252 trading days in a year
            return annual_return / (max_drawdown + 1e-10)
        except Exception as e:
            logger.error(f"Error calculating Calmar Ratio: {e}")
            return None

    def calculate_omega_ratio(self, returns, risk_free_rate=0.02, target_return=0):
        try:
            returns = np.array(returns)
            excess_returns = returns - risk_free_rate
            positive_returns = excess_returns[excess_returns > target_return]
            negative_returns = excess_returns[excess_returns <= target_return]
            return (np.sum(positive_returns) + 1e-10) / (abs(np.sum(negative_returns)) + 1e-10)
        except Exception as e:
            logger.error(f"Error calculating Omega Ratio: {e}")
            return None

    # ... (rest of the class remains the same)
