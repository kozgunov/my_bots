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
        
        # Initialize metrics
        self.reset_metrics()

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
        if not self.trade_history or self.initial_balance is None or self.initial_balance == 0:
            return {
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'total_return': 0,
                'win_rate': 0,
                'weighted_win_rate': 0,
                'profit_factor': 0,
                'avg_profit_per_trade': 0,
                'risk_reward_ratio': 0
            }

        profits = [trade[1] for trade in self.trade_history if trade[1] > 0]
        losses = [abs(trade[1]) for trade in self.trade_history if trade[1] < 0]
        
        total_trades = len(self.trade_history)
        winning_trades = len(profits)
        
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
            drawdown = (peak - balance) / peak if peak > 0 else 0
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
        total_return = (self.current_balance - self.initial_balance) / self.initial_balance if self.initial_balance > 0 else 0

        return {
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'total_return': total_return,
            'win_rate': winning_trades / total_trades if total_trades > 0 else 0,
            'weighted_win_rate': weighted_win_rate,
            'profit_factor': profit_factor,
            'avg_profit_per_trade': avg_profit_per_trade,
            'risk_reward_ratio': risk_reward_ratio
        }

def train_agent(env, agent, episodes, batch_size=64, update_target_every=5, checkpoint_every=100, early_stopping_patience=50):
    print("You are here: Starting training process")
    rewards_history = []
    avg_rewards_history = []
    epsilon_history = []
    loss_history = []
    metrics_history = []
    early_terminations = 0

    best_score = -np.inf
    best_model_path = None
    best_metrics = None
    no_improvement_count = 0

    results_dir = os.path.join(os.path.expanduser("~"), "Desktop", "binance_data")
    os.makedirs(results_dir, exist_ok=True)

    for episode in range(episodes):
        print(f"You are here: Starting episode {episode}")
        total_reward = 0
        state = env.reset()
        agent.reset_metrics()

        done = False
        episode_losses = []
        while not done:
            action = agent.get_action(state)
            next_state, reward, done, info = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            agent.update_metrics(action, reward, env.balance)
            state = next_state
            total_reward += reward

            loss = agent.replay(batch_size)
            if loss is not None:
                episode_losses.append(loss)

            if info.get("early_termination", False):
                early_terminations += 1

        if episode % update_target_every == 0:
            agent.update_target_model()

        rewards_history.append(total_reward)
        avg_reward = np.mean(rewards_history[-100:])
        avg_rewards_history.append(avg_reward)
        epsilon_history.append(agent.epsilon)

        # Calculate average loss for the episode
        avg_loss = np.mean(episode_losses) if episode_losses else 0
        loss_history.append(avg_loss)

        metrics = agent.calculate_metrics()
        metrics_history.append(metrics)

        print(f"Episode {episode} completed:")
        print(f"  Total Reward: {total_reward:.2f}")
        print(f"  Avg Reward: {avg_reward:.2f}")
        print(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        print(f"  Max Drawdown: {metrics['max_drawdown']:.2%}")
        print(f"  Weighted Win Rate: {metrics['weighted_win_rate']:.2%}")
        print(f"  Epsilon: {agent.epsilon:.4f}")
        print(f"  Avg Loss: {avg_loss:.4f}")
        print(f"  Early Terminations: {early_terminations}")

        # Calculate a composite score for the model's performance
        score = (
            metrics['sharpe_ratio'] * 0.3 +
            metrics['weighted_win_rate'] * 0.3 +
            (1 - metrics['max_drawdown']) * 0.2 +
            avg_reward * 0.2
        )

        if score > best_score:
            best_score = score
            best_model_path = f"best_model_episode_{episode}.pth"
            agent.save(best_model_path)
            best_metrics = metrics.copy()
            print(f"New best model saved with score: {best_score:.2f}")
            no_improvement_count = 0
        else:
            no_improvement_count += 1

        # Adjust epsilon decay based on performance
        if metrics['weighted_win_rate'] < 0.5:
            agent.epsilon_decay = max(agent.epsilon_decay, 0.995)  # Slow down decay if win rate is low
        else:
            agent.epsilon_decay = min(agent.epsilon_decay, 0.99)  # Speed up decay if win rate is high

        if no_improvement_count >= early_stopping_patience:
            print(f"Early stopping at episode {episode}")
            break

        if episode % checkpoint_every == 0 or episode == episodes - 1:
            plot_training_results(rewards_history, avg_rewards_history, epsilon_history, loss_history, metrics_history, results_dir, episode, early_terminations)

    print("You are here: Training completed")
    final_results_dir = os.path.join(results_dir, "Final_results")
    os.makedirs(final_results_dir, exist_ok=True)
    plot_training_results(rewards_history, avg_rewards_history, epsilon_history, loss_history, metrics_history, final_results_dir, episodes, early_terminations)

    if best_model_path:
        shutil.copy(best_model_path, os.path.join(final_results_dir, 'best_model.pth'))

    print_final_metrics(best_metrics, final_results_dir, early_terminations)

    return best_model_path, best_metrics

def plot_training_results(rewards, avg_rewards, epsilons, losses, metrics_history, results_dir, episode, early_terminations):
    print(f"You are here: Plotting results for episode {episode}")
    
    # Create a subdirectory for this episode's plots
    episode_dir = os.path.join(results_dir, f'episode_{episode}')
    os.makedirs(episode_dir, exist_ok=True)

    # Plot rewards
    plt.figure(figsize=(10, 6))
    plt.plot(rewards, label='Reward', alpha=0.6)
    plt.plot(avg_rewards, label='100-episode Average', color='red')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Training Rewards')
    plt.legend()
    plt.savefig(os.path.join(episode_dir, 'rewards.png'))
    plt.close()

    # Plot epsilon decay
    plt.figure(figsize=(10, 6))
    plt.plot(epsilons)
    plt.xlabel('Episode')
    plt.ylabel('Epsilon')
    plt.title('Epsilon Decay')
    plt.savefig(os.path.join(episode_dir, 'epsilon_decay.png'))
    plt.close()

    # Plot training loss
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.savefig(os.path.join(episode_dir, 'training_loss.png'))
    plt.close()

    # Plot weighted win rate
    plt.figure(figsize=(10, 6))
    plt.plot([m['weighted_win_rate'] for m in metrics_history])
    plt.xlabel('Episode')
    plt.ylabel('Weighted Win Rate')
    plt.title('Weighted Win Rate')
    plt.savefig(os.path.join(episode_dir, 'weighted_win_rate.png'))
    plt.close()

    # Plot Sharpe ratio
    plt.figure(figsize=(10, 6))
    plt.plot([m['sharpe_ratio'] for m in metrics_history])
    plt.xlabel('Episode')
    plt.ylabel('Sharpe Ratio')
    plt.title('Sharpe Ratio')
    plt.savefig(os.path.join(episode_dir, 'sharpe_ratio.png'))
    plt.close()

    # Plot maximum drawdown
    plt.figure(figsize=(10, 6))
    plt.plot([m['max_drawdown'] for m in metrics_history])
    plt.xlabel('Episode')
    plt.ylabel('Maximum Drawdown')
    plt.title('Maximum Drawdown')
    plt.savefig(os.path.join(episode_dir, 'max_drawdown.png'))
    plt.close()

    # Plot profit factor
    plt.figure(figsize=(10, 6))
    plt.plot([m['profit_factor'] for m in metrics_history])
    plt.xlabel('Episode')
    plt.ylabel('Profit Factor')
    plt.title('Profit Factor')
    plt.savefig(os.path.join(episode_dir, 'profit_factor.png'))
    plt.close()

    # Plot average profit per trade
    plt.figure(figsize=(10, 6))
    plt.plot([m['avg_profit_per_trade'] for m in metrics_history])
    plt.xlabel('Episode')
    plt.ylabel('Average Profit per Trade')
    plt.title('Average Profit per Trade')
    plt.savefig(os.path.join(episode_dir, 'avg_profit_per_trade.png'))
    plt.close()

    # Plot risk-reward ratio
    plt.figure(figsize=(10, 6))
    plt.plot([m['risk_reward_ratio'] for m in metrics_history])
    plt.xlabel('Episode')
    plt.ylabel('Risk-Reward Ratio')
    plt.title('Risk-Reward Ratio')
    plt.savefig(os.path.join(episode_dir, 'risk_reward_ratio.png'))
    plt.close()

    # Plot total return
    plt.figure(figsize=(10, 6))
    plt.plot([m['total_return'] for m in metrics_history])
    plt.xlabel('Episode')
    plt.ylabel('Total Return')
    plt.title('Total Return')
    plt.savefig(os.path.join(episode_dir, 'total_return.png'))
    plt.close()

    # Plot early termination rate
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(rewards)), [early_terminations / (i+1) for i in range(len(rewards))])
    plt.xlabel('Episode')
    plt.ylabel('Early Termination Rate')
    plt.title('Early Termination Rate')
    plt.savefig(os.path.join(episode_dir, 'early_termination_rate.png'))
    plt.close()

    print(f"Plots saved in {episode_dir}")

def print_final_metrics(final_metrics, final_results_dir, early_terminations):
    print("\nFinal Performance Metrics:")
    for key, value in final_metrics.items():
        print(f"{key}: {value:.4f}")
    print(f"Total Early Terminations: {early_terminations}")

    with open(os.path.join(final_results_dir, 'final_metrics.txt'), 'w') as f:
        for key, value in final_metrics.items():
            f.write(f"{key}: {value:.4f}\n")
        f.write(f"Total Early Terminations: {early_terminations}\n")

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
    state_size = 4  # balance, holdings, current price, time since last action
    agent = ImprovedAgent(
        state_size=state_size, 
        action_size=3,
        learning_rate=0.0005,  # Reduced learning rate
        gamma=0.99,  # Increased discount factor
        epsilon=1.0,
        epsilon_decay=0.9995,  # Slower epsilon decay
        epsilon_min=0.05,
        memory_size=10000,  # Increased memory size
        batch_size=64
    )

    # Train the agent
    print("You are here: Starting training")
    train_agent(env, agent, EPISODES, batch_size=64)

    print("You are here: Training completed. Final results saved in 'Final_results' folder.")

