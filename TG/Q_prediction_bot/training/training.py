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
import ccxt
import json
import sys

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

# Force CUDA to be the default device type
torch.set_default_tensor_type('torch.cuda.FloatTensor')

# Enable cudnn benchmark for improved performance
torch.backends.cudnn.benchmark = True

class SimplerDQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SimplerDQN, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Linear(32, output_dim)
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
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=5, factor=0.5, verbose=True)

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*minibatch)
        
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        current_q_values = self.model(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.target_model(next_states).max(1)[0].detach()
        target_q_values = rewards + (self.gamma * next_q_values * (1 - dones))

        # Target clipping
        target_q_values = torch.clamp(target_q_values, min=-1, max=1)

        loss = nn.MSELoss()(current_q_values, target_q_values.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        return loss.item()

def train_agent(env, agent, episodes, batch_size=64, update_target_every=5, checkpoint_every=100, early_stopping_patience=50):
    print("You are here: Starting training process")
    rewards_history = []
    avg_rewards_history = []
    epsilon_history = []
    loss_history = []
    win_history = []
    best_avg_reward = -np.inf
    no_improvement_count = 0

    # Create results directory
    results_dir = os.path.join(os.path.expanduser("~"), "Desktop", "results")
    os.makedirs(results_dir, exist_ok=True)

    # Create a dataset and dataloader for the environment states
    states = torch.FloatTensor([env._get_state() for _ in range(len(env.data))]).to(device)
    dataset = TensorDataset(states)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True)

    accumulation_steps = 4  # Adjust as needed
    for episode in range(episodes):
        print(f"You are here: Starting episode {episode}")
        total_reward = 0
        wins = 0
        losses = 0
        state = env.reset()

        for i, batch in enumerate(dataloader):
            batch_states = batch[0]
            
            actions = agent.get_action(batch_states)
            next_states, rewards, dones = [], [], []

            for j, action in enumerate(actions):
                next_state, reward, done = env.step(action.item())
                next_states.append(next_state)
                rewards.append(reward)
                dones.append(done)
                total_reward += reward

                # Count wins and losses
                if reward > 0:
                    wins += 1
                elif reward < 0:
                    losses += 1

                if done:
                    break

            next_states = torch.FloatTensor(next_states).to(device)
            rewards = torch.FloatTensor(rewards).to(device)
            dones = torch.FloatTensor(dones).to(device)

            agent.remember(batch_states, actions, rewards, next_states, dones)
            loss = agent.replay(batch_size)
            loss = loss / accumulation_steps
            loss.backward()
            
            if (i + 1) % accumulation_steps == 0:
                agent.optimizer.step()
                agent.optimizer.zero_grad()
            
            loss_history.append(loss.item())
            agent.scheduler.step(loss)

        if episode % update_target_every == 0:
            agent.update_target_model()

        rewards_history.append(total_reward)
        avg_reward = np.mean(rewards_history[-100:])
        avg_rewards_history.append(avg_reward)
        epsilon_history.append(agent.epsilon)
        win_history.append(wins / (wins + losses) if wins + losses > 0 else 0)

        if episode % checkpoint_every == 0:
            agent.save(f"model_checkpoint_episode_{episode}.pth")
            print(f"You are here: Checkpoint saved at episode {episode}")

        if avg_reward > best_avg_reward:
            best_avg_reward = avg_reward
            agent.save("best_model.pth")
            no_improvement_count = 0
        else:
            no_improvement_count += 1

        if no_improvement_count >= early_stopping_patience:
            print(f"You are here: Early stopping at episode {episode}")
            break

        if episode % 10 == 0:
            print(f"Episode: {episode}, Total Reward: {total_reward:.2f}, Avg Reward: {avg_reward:.2f}, Epsilon: {agent.epsilon:.2f}, Win Rate: {win_history[-1]:.2f}")
            plot_training_results(rewards_history, avg_rewards_history, epsilon_history, loss_history, win_history, results_dir, episode, agent)

    print("You are here: Training completed")
    plot_training_results(rewards_history, avg_rewards_history, epsilon_history, loss_history, win_history, results_dir, episodes, agent)

def plot_training_results(rewards, avg_rewards, epsilons, losses, win_rates, results_dir, episode, agent):
    print(f"You are here: Plotting results for episode {episode}")
    plt.figure(figsize=(20, 25))

    # Reward plot
    plt.subplot(5, 2, 1)
    plt.plot(rewards, label='Reward', alpha=0.6)
    plt.plot(avg_rewards, label='100-episode Average', color='red')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Training Rewards')
    plt.legend()

    # Smoothed reward plot
    plt.subplot(5, 2, 2)
    window_size = 100
    smoothed_rewards = pd.Series(rewards).rolling(window=window_size).mean()
    plt.plot(smoothed_rewards, label=f'{window_size}-episode Moving Average', color='green')
    plt.xlabel('Episode')
    plt.ylabel('Smoothed Reward')
    plt.title(f'Smoothed Rewards ({window_size}-episode window)')
    plt.legend()

    # Epsilon decay
    plt.subplot(5, 2, 3)
    plt.plot(epsilons)
    plt.xlabel('Episode')
    plt.ylabel('Epsilon')
    plt.title('Epsilon Decay')

    # Loss plot
    plt.subplot(5, 2, 4)
    plt.plot(losses)
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.title('Training Loss')

    # Win rate plot
    plt.subplot(5, 2, 5)
    plt.plot(win_rates, color='purple')
    plt.xlabel('Episode')
    plt.ylabel('Win Rate')
    plt.title('Win Rate per Episode')

    # Correlation heatmap
    plt.subplot(5, 2, 6)
    corr_data = pd.DataFrame({'reward': rewards, 'avg_reward': avg_rewards, 'epsilon': epsilons, 'win_rate': win_rates})
    sns.heatmap(corr_data.corr(), annot=True, cmap='coolwarm')
    plt.title('Correlation Heatmap')

    # Reward distribution
    plt.subplot(5, 2, 7)
    sns.histplot(rewards, kde=True)
    plt.xlabel('Reward')
    plt.title('Reward Distribution')

    # Q-value distribution (if available)
    if hasattr(agent, 'model'):
        plt.subplot(5, 2, 8)
        # Generate some sample states for Q-value distribution
        sample_states = np.random.rand(1000, agent.state_size)
        q_values = agent.model(torch.FloatTensor(sample_states).to(agent.device)).detach().cpu().numpy()
        sns.histplot(q_values.flatten(), kde=True)
        plt.xlabel('Q-value')
        plt.title('Q-value Distribution (1000 random states)')

    # Learning rate plot
    plt.subplot(5, 2, 9)
    lr_history = [group['lr'] for group in agent.optimizer.param_groups]
    plt.plot(lr_history)
    plt.xlabel('Step')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Decay')

    # Cumulative reward plot
    plt.subplot(5, 2, 10)
    cumulative_rewards = np.cumsum(rewards)
    plt.plot(cumulative_rewards)
    plt.xlabel('Episode')
    plt.ylabel('Cumulative Reward')
    plt.title('Cumulative Reward')

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f'training_results_episode_{episode}.png'))
    plt.close()

    # Save data to CSV
    pd.DataFrame({
        'episode': range(len(rewards)),
        'reward': rewards,
        'avg_reward': avg_rewards,
        'epsilon': epsilons,
        'loss': losses[:len(rewards)],  # Trim losses to match rewards length
        'win_rate': win_rates
    }).to_csv(os.path.join(results_dir, f'training_data_episode_{episode}.csv'), index=False)

def save_binance_data(symbol, timeframe='10m', limit=5000):
    """
    Save Binance data locally and implement a safety check.
    
    Args:
    symbol (str): Trading pair symbol (e.g., 'BTC/USDT')
    timeframe (str): Data timeframe (e.g., '10m' for 10 minutes)
    limit (int): Number of candles to fetch
    
    Returns:
    pandas.DataFrame: DataFrame with the fetched data
    """
    try:
        exchange = ccxt.binance()
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        
        # Save data locally
        data_dir = os.path.join(os.path.expanduser("~"), "Desktop", "binance_data")
        os.makedirs(data_dir, exist_ok=True)
        file_name = f"{symbol.replace('/', '_')}_{timeframe}_{limit}.csv"
        file_path = os.path.join(data_dir, file_name)
        df.to_csv(file_path)
        
        print(f"Data saved successfully to {file_path}")
        return df
    except Exception as e:
        print(f"Error fetching or saving Binance data: {e}")
        sys.exit(1)  # Stop the program if something goes wrong

if __name__ == "__main__":
    print("You are here: Starting main function")
    
    # Save Binance data locally
    print("You are here: Saving Binance data locally")
    df = save_binance_data(SYMBOL, timeframe='10m', limit=DATA_LIMIT)
    
    if df is None:
        print("Failed to load data. Exiting.")
        sys.exit(1)

    # Prepare data
    print("You are here: Preparing data")
    df, scaler = prepare_data('binance', SYMBOL, timeframe='10m', limit=DATA_LIMIT)

    # Create sequences
    print("You are here: Creating sequences")
    seq_length = 144  # 24 hours of data (144 * 10 minutes = 24 hours)
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
    agent = ImprovedAgent(
        state_size=X.shape[1] * X.shape[2] + 2, 
        action_size=3,
        learning_rate=LEARNING_RATE, 
        gamma=GAMMA, 
        epsilon=EPSILON, 
        epsilon_decay=EPSILON_DECAY, 
        epsilon_min=EPSILON_MIN, 
        memory_size=MEMORY_SIZE, 
        batch_size=BATCH_SIZE
    ).to(device)  # Ensure the agent is on the GPU

    # Train the agent
    print("You are here: Starting training")
    train_agent(env, agent, EPISODES, batch_size=BATCH_SIZE)

    print("You are here: Training completed. Model saved as 'best_model.pth'")
