
import numpy as np
import matplotlib.pyplot as plt
from environment import TradingEnvironment
from agent import ImprovedQLearningAgent
from utils import prepare_data, create_sequences
from config import *
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim 
from torch.utils.data import DataLoader, TensorDataset
import random 
import os
import pandas as pd
import seaborn as sns
import sys
import shutil

# Check CUDA availability and print device info
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device count: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"Current CUDA device: {torch.cuda.current_device()}")
    print(f"CUDA device name: {torch.cuda.get_device_name(0)}")

# Ensure deterministic behavior
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(0)
torch.manual_seed(0)
random.seed(0)

# Check if CUDA is available and force GPU usage
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cpu":
    print("WARNING: No GPU detected. Training will be slow on CPU.")
    print("If you have a GPU, please ensure CUDA is properly installed and configured.")
else:
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")

# Enable cudnn benchmark for improved performance
torch.backends.cudnn.benchmark = True

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

class ImprovedAgent(ImprovedQLearningAgent):
    def __init__(self, state_size, action_size, learning_rate=0.001, gamma=0.95, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01, memory_size=2000, batch_size=32):
        super().__init__(state_size, action_size, learning_rate, gamma, epsilon, epsilon_decay, epsilon_min, memory_size, batch_size)
        self.device = device
        self.model = SimplerDQN(state_size, action_size).to(self.device)
        self.target_model = SimplerDQN(state_size, action_size).to(self.device)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=5, factor=0.5)
        
        # New attributes for advanced metrics
        self.trade_history = []
        self.initial_balance = None
        self.current_balance = None
        self.max_balance = None

    def reset_metrics(self):
        self.trade_history = []
        self.initial_balance = None
        self.current_balance = None
        self.max_balance = None

    def update_metrics(self, action, reward, balance):
        if self.initial_balance is None:
            self.initial_balance = balance
            self.max_balance = balance
        self.current_balance = balance
        self.max_balance = max(self.max_balance, balance)
        self.trade_history.append((action, reward, balance))

    def calculate_metrics(self):
        if not self.trade_history:
            return {}

        profits = [trade[1] for trade in self.trade_history if trade[1] > 0]
        losses = [abs(trade[1]) for trade in self.trade_history if trade[1] < 0]
        
        total_trades = len(self.trade_history)
        winning_trades = len(profits)
        
        if total_trades == 0:
            return {}

        # Weighted Win Rate
        weighted_win_rate = sum(profits) / (sum(profits) + sum(losses)) if (sum(profits) + sum(losses)) > 0 else 0

        # Sharpe Ratio (assuming risk-free rate of 0 for simplicity)
        returns = [trade[1] / self.initial_balance for trade in self.trade_history]
        sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0

        # Maximum Drawdown
        peak = self.initial_balance
        max_drawdown = 0
        for _, _, balance in self.trade_history:
            if balance > peak:
                peak = balance
            drawdown = (peak - balance) / peak
            max_drawdown = max(max_drawdown, drawdown)

        # Profit Factor
        profit_factor = sum(profits) / sum(losses) if sum(losses) > 0 else float('inf')

        # Average Profit per Trade
        avg_profit_per_trade = np.mean([trade[1] for trade in self.trade_history])

        # Risk-Reward Ratio (using average win and average loss as proxies)
        avg_win = np.mean(profits) if profits else 0
        avg_loss = np.mean(losses) if losses else 0
        risk_reward_ratio = avg_win / avg_loss if avg_loss > 0 else float('inf')

        # Total Return
        total_return = (self.current_balance - self.initial_balance) / self.initial_balance

        return {
            'weighted_win_rate': weighted_win_rate,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'profit_factor': profit_factor,
            'avg_profit_per_trade': avg_profit_per_trade,
            'risk_reward_ratio': risk_reward_ratio,
            'total_return': total_return,
            'win_rate': winning_trades / total_trades
        }

def train_agent(env, agent, episodes, batch_size=64, update_target_every=5, checkpoint_every=100, early_stopping_patience=50):
    print("You are here: Starting training process")
    rewards_history = []
    avg_rewards_history = []
    epsilon_history = []
    loss_history = []
    metrics_history = []

    best_sharpe_ratio = -np.inf
    best_model_path = None
    no_improvement_count = 0

    results_dir = os.path.join(os.path.expanduser("~"), "Desktop", "binance_data")
    os.makedirs(results_dir, exist_ok=True)

    for episode in range(episodes):
        print(f"You are here: Starting episode {episode}")
        total_reward = 0
        state = env.reset()
        agent.reset_metrics()

        done = False
        while not done:
            action = agent.get_action(state)
            next_state, reward, done = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            agent.update_metrics(action, reward, env.balance)
            state = next_state
            total_reward += reward

            loss = agent.replay(batch_size)
            if loss is not None:
                loss_history.append(loss)

        if episode % update_target_every == 0:
            agent.update_target_model()

        rewards_history.append(total_reward)
        avg_reward = np.mean(rewards_history[-100:])
        avg_rewards_history.append(avg_reward)
        epsilon_history.append(agent.epsilon)

        metrics = agent.calculate_metrics()
        metrics_history.append(metrics)

        print(f"Episode {episode} completed: Total Reward: {total_reward:.2f}, Sharpe Ratio: {metrics['sharpe_ratio']:.2f}, Max Drawdown: {metrics['max_drawdown']:.2%}")

        if metrics['sharpe_ratio'] > best_sharpe_ratio:
            best_sharpe_ratio = metrics['sharpe_ratio']
            best_model_path = f"best_model_episode_{episode}.pth"
            agent.save(best_model_path)
            print(f"New best Sharpe Ratio: {best_sharpe_ratio:.2f}")
            no_improvement_count = 0
        else:
            no_improvement_count += 1

        if no_improvement_count >= early_stopping_patience:
            print(f"Early stopping at episode {episode}")
            break

        if episode % checkpoint_every == 0 or episode == episodes - 1:
            plot_training_results(rewards_history, avg_rewards_history, epsilon_history, loss_history, metrics_history, results_dir, episode)

    print("You are here: Training completed")
    final_results_dir = os.path.join(results_dir, "Final_results")
    os.makedirs(final_results_dir, exist_ok=True)
    plot_training_results(rewards_history, avg_rewards_history, epsilon_history, loss_history, metrics_history, final_results_dir, episodes)

    if best_model_path:
        shutil.copy(best_model_path, os.path.join(final_results_dir, 'best_model.pth'))

    print_final_metrics(metrics_history[-1], final_results_dir)

def plot_training_results(rewards, avg_rewards, epsilons, losses, metrics_history, results_dir, episode):
    print(f"You are here: Plotting results for episode {episode}")
    
    plt.figure(figsize=(20, 30))

    # Existing plots
    plt.subplot(6, 2, 1)
    plt.plot(rewards, label='Reward', alpha=0.6)
    plt.plot(avg_rewards, label='100-episode Average', color='red')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Training Rewards')
    plt.legend()

    plt.subplot(6, 2, 2)
    plt.plot(epsilons)
    plt.xlabel('Episode')
    plt.ylabel('Epsilon')
    plt.title('Epsilon Decay')

    plt.subplot(6, 2, 3)
    plt.plot(losses)
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.title('Training Loss')

    # New plots for advanced metrics
    plt.subplot(6, 2, 4)
    plt.plot([m['weighted_win_rate'] for m in metrics_history])
    plt.xlabel('Episode')
    plt.ylabel('Weighted Win Rate')
    plt.title('Weighted Win Rate')

    plt.subplot(6, 2, 5)
    plt.plot([m['sharpe_ratio'] for m in metrics_history])
    plt.xlabel('Episode')
    plt.ylabel('Sharpe Ratio')
    plt.title('Sharpe Ratio')

    plt.subplot(6, 2, 6)
    plt.plot([m['max_drawdown'] for m in metrics_history])
    plt.xlabel('Episode')
    plt.ylabel('Maximum Drawdown')
    plt.title('Maximum Drawdown')

    plt.subplot(6, 2, 7)
    plt.plot([m['profit_factor'] for m in metrics_history])
    plt.xlabel('Episode')
    plt.ylabel('Profit Factor')
    plt.title('Profit Factor')

    plt.subplot(6, 2, 8)
    plt.plot([m['avg_profit_per_trade'] for m in metrics_history])
    plt.xlabel('Episode')
    plt.ylabel('Average Profit per Trade')
    plt.title('Average Profit per Trade')

    plt.subplot(6, 2, 9)
    plt.plot([m['risk_reward_ratio'] for m in metrics_history])
    plt.xlabel('Episode')
    plt.ylabel('Risk-Reward Ratio')
    plt.title('Risk-Reward Ratio')

    plt.subplot(6, 2, 10)
    plt.plot([m['total_return'] for m in metrics_history])
    plt.xlabel('Episode')
    plt.ylabel('Total Return')
    plt.title('Total Return')

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f'training_results_episode_{episode}.png'))
    plt.close()

def print_final_metrics(final_metrics, final_results_dir):
    print("\nFinal Performance Metrics:")
    for key, value in final_metrics.items():
        print(f"{key}: {value:.4f}")

    with open(os.path.join(final_results_dir, 'final_metrics.txt'), 'w') as f:
        for key, value in final_metrics.items():
            f.write(f"{key}: {value:.4f}\n")

def load_local_data(file_path):
    """
    Load data from a local CSV file.
    
    Args:
    file_path (str): Path to the CSV file
    
    Returns:
    pandas.DataFrame: DataFrame with the loaded data
    """
    try:
        df = pd.read_csv(file_path, index_col='timestamp', parse_dates=True)
        print(f"Data loaded successfully from {file_path}")
        print(f"Data range: from {df.index.min()} to {df.index.max()}")
        print(f"Total days: {(df.index.max() - df.index.min()).days}")
        return df
    except Exception as e:
        print(f"Error loading data from {file_path}: {e}")
        return None

if __name__ == "__main__":
    print("You are here: Starting main function")
    
    # Load data from local CSV file
    print("You are here: Loading data from local CSV file")
    file_path = "C:/Users/user/pythonProject/education/BTC_USDT_15m_4000.csv"
    df = load_local_data(file_path)
    
    if df is None or len(df) < 1000:  # Adjust this threshold as needed
        print("Failed to load sufficient data. Exiting.")
        sys.exit(1)

    # Prepare data
    print("You are here: Preparing data")
    df, scaler = prepare_data(df)

    if df is None or len(df) < 1000:
        print("Failed to prepare sufficient data. Exiting.")
        sys.exit(1)

    # Create sequences
    print("You are here: Creating sequences")
    seq_length = 96  # 24 hours of data (96 * 15 minutes = 24 hours)
    X, y = create_sequences(df, seq_length)
    y = y[:, df.columns.get_loc('close')]  # We only want to predict the 'close' price

    # Use more data for training
    print("You are here: Splitting data")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    # Create environment with training data only
    print("You are here: Creating environment")
    env = TradingEnvironment(y_train, initial_balance=INITIAL_BALANCE, fee=TRADING_FEE)

    # Create an improved agent
    print("You are here: Creating agent")
    state_size = 3  # balance, holdings, current price
    agent = ImprovedAgent(
        state_size=state_size, 
        action_size=3,
        learning_rate=LEARNING_RATE, 
        gamma=GAMMA, 
        epsilon=EPSILON, 
        epsilon_decay=EPSILON_DECAY, 
        epsilon_min=EPSILON_MIN, 
        memory_size=MEMORY_SIZE, 
        batch_size=BATCH_SIZE
    )

    # Train the agent
    print("You are here: Starting training")
    train_agent(env, agent, EPISODES, batch_size=BATCH_SIZE)

    print("You are here: Training completed. Final results saved in 'Final_results' folder.")
