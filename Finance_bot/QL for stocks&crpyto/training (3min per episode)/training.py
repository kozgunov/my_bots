import numpy as np
import matplotlib.pyplot as plt
from environment import TradingEnvironment
from agent import ImprovedQLearningAgent
from utils import calculate_sharpe_ratio, calculate_max_drawdown, calculate_profit_factor, plot_shap_values
from config import *
import torch
import os
import pandas as pd
import logging
import shap
import json
import time
import random
import sys


# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Disable debug logging for other modules
logging.getLogger("agent").setLevel(logging.INFO)
logging.getLogger("environment").setLevel(logging.INFO)

# Get the path to the desktop
DESKTOP_PATH = os.path.join(os.path.expanduser("~"), "Desktop")
TRAINING_RESULTS_PATH = os.path.join(DESKTOP_PATH, "training_results")
TESTING_RESULTS_PATH = os.path.join(DESKTOP_PATH, "testing_results")

# Create the results folders if they don't exist
os.makedirs(TRAINING_RESULTS_PATH, exist_ok=True)
os.makedirs(TESTING_RESULTS_PATH, exist_ok=True)

def load_json_data(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # Remove timestamps
    for item in data:
        item.pop('timestamp', None)
    
    return pd.DataFrame(data)

def plot_best_model_results(agent, episode, rewards, epsilons, win_rates, losses):
    # Create a directory for best model results
    best_model_dir = os.path.join(TRAINING_RESULTS_PATH, f"best_model_episode_{episode}")
    os.makedirs(best_model_dir, exist_ok=True)

    # Plot training results
    plot_training_results(rewards, epsilons, win_rates, episode, save_path=best_model_dir)
    
    # Plot 3D metrics
    plot_3d_metrics(rewards, win_rates, losses, episode, save_path=best_model_dir)
    
    # Plot individual plots
    plot_individual_plots(rewards, epsilons, win_rates, losses, episode, save_path=best_model_dir)
    
    # Plot action distribution
    plot_action_distribution(agent, episode, save_path=best_model_dir)
    
    # Generate and save SHAP plot
    try:
        sample_data = agent.memory.buffer[-100:]  # Use the last 100 experiences
        states = np.array([exp[0] for exp in sample_data])
        plot_shap_values(agent.model, states, agent.feature_names, best_model_dir, episode)
    except Exception as e:
        logger.error(f"Error in generating SHAP plot for best model: {e}")

    # Add gradient descent plot
    plot_gradient_descent(agent, episode, save_path=best_model_dir)

def plot_action_distribution(agent, episode, save_path):
    action_dist = agent.get_action_distribution()
    plt.figure(figsize=(10, 6))
    plt.bar(action_dist.keys(), action_dist.values())
    plt.title(f'Action Distribution - Episode {episode}')
    plt.xlabel('Action')
    plt.ylabel('Frequency')
    plt.savefig(os.path.join(save_path, f"action_distribution_{episode}.png"))
    plt.close()

def train_agent(env, agent, episodes, batch_size=64, checkpoint_interval=10, patience=50, max_time_per_episode=180):
    logger.info(f"Starting training for {episodes} episodes")
    best_score = -np.inf
    no_improvement_count = 0
    all_rewards = []
    all_epsilons = []
    all_win_rates = []
    all_losses = []  # Keep track of losses for monitoring

    for episode in range(episodes):
        state = env.reset()
        done = False
        episode_reward = 0
        episode_steps = 0
        episode_trades = 0
        episode_wins = 0
        actions = []
        price_data = []
        episode_losses = []  # Track losses for this episode
        start_time = time.time()

        buy_signals = []
        sell_signals = []
        long_positions = 0
        short_positions = 0

        while not done and time.time() - start_time < max_time_per_episode:
            try:
                action = agent.act(state)
                next_state, reward, done, info = env.step(action)
                agent.remember(state, action, reward, next_state, done)
                state = next_state
                episode_reward += reward
                episode_steps += 1

                actions.append(action)
                if 'current_price' in info:
                    price_data.append(info['current_price'])
                else:
                    logger.warning(f"'current_price' not found in info at step {episode_steps}")

                if info.get('trade_made', False):
                    episode_trades += 1
                    if reward > 0:
                        episode_wins += 1
                    else: 
                        episode_wins -= 1

                if action == 1:
                    buy_signals.append(episode_steps)
                    long_positions += 1
                elif action == 2:
                    sell_signals.append(episode_steps)
                    short_positions += 1

                loss = agent.replay(batch_size)
                if loss is not None:
                    episode_losses.append(loss)
            except Exception as e:
                logger.error(f"Error during episode {episode} step: {e}")
                break

        # Calculate win rate and weighted win rate
        if episode_trades > 0:
            win_rate = episode_wins / episode_trades
            trade_frequency = episode_trades / episode_steps
            weighted_win_rate = win_rate * trade_frequency
        else:
            win_rate = 0
            weighted_win_rate = 0

        all_rewards.append(episode_reward)
        all_epsilons.append(agent.epsilon)
        all_win_rates.append(weighted_win_rate)
        all_losses.append(np.mean(episode_losses) if episode_losses else 0)  # Average loss for the episode

        # Score is now based only on reward and win rate
        score = episode_reward + (weighted_win_rate * 100)
        
        logger.info(f"Episode {episode+1}/{episodes} - Score: {score:.2f}, Reward: {episode_reward:.2f}, "
                    f"Win Rate: {win_rate:.2f}, Weighted Win Rate: {weighted_win_rate:.4f}, "
                    f"Trades: {episode_trades}, Steps: {episode_steps}, Avg Loss: {all_losses[-1]:.4f}")

        agent.decay_epsilon()

        if score > best_score:
            best_score = score
            agent.save(os.path.join(TRAINING_RESULTS_PATH, f"model_best_score_{episode}.pth"))
            save_model_details(agent, episode, score, weighted_win_rate, episode_reward, actions, all_losses[-1])
            plot_best_model_results(agent, episode, all_rewards, all_epsilons, all_win_rates, all_losses)
            
            # SHAP analysis for best model
            try:
                sample_data = agent.memory.buffer[-100:]  # Use the last 100 experiences
                states = np.array([exp[0] for exp in sample_data])
                shap_values, expected_value = plot_shap_values(agent.model, states, agent.feature_names, TRAINING_RESULTS_PATH, episode)
                
                if shap_values is not None and expected_value is not None:
                    analyze_shap_values(shap_values, expected_value, agent.feature_names, episode)
            except Exception as e:
                logger.error(f"Error in SHAP analysis for best model: {e}")

            logger.info(f"New best model (score) saved at episode {episode+1}")
            no_improvement_count = 0
        else:
            no_improvement_count += 1

        if episode % checkpoint_interval == 0 or episode == episodes - 1:
            agent.save(os.path.join(TRAINING_RESULTS_PATH, f"model_checkpoint_{episode}.pth"))
            try:
                plot_individual_plots(all_rewards, all_epsilons, all_win_rates, all_losses, episode)
                plot_3d_metrics(all_rewards, all_win_rates, all_losses, episode)
                if price_data:
                    plot_trading_decisions(price_data, actions, episode, buy_signals, sell_signals)
                else:
                    logger.warning(f"No price data available for plotting trading decisions in episode {episode}")
                plot_position_openings(long_positions, short_positions, episode)
                plot_gradient_descent(agent, episode)  # Add this line
                logger.info(f"Checkpoint saved and plots generated at episode {episode+1}")
            except Exception as e:
                logger.error(f"Error during plotting for episode {episode}: {e}")

        if no_improvement_count >= patience:
            logger.info(f"Stopping early due to no improvement for {patience} episodes")
            break

        episode_time = time.time() - start_time
        logger.info(f"Episode {episode+1} duration: {episode_time:.2f} seconds")

    return agent, all_rewards, all_epsilons, all_win_rates, all_losses

def evaluate_agent(env, agent, episodes=100):
    all_rewards = []
    all_win_rates = []
    all_losses = []
    total_trades = 0
    winning_trades = 0
    buy_trades = 0
    sell_trades = 0
    buy_wins = 0
    sell_wins = 0
    total_trade_amount = 0
    weighted_win_sum = 0

    portfolio_values = []
    returns = []

    for episode in range(episodes):
        state = env.reset()
        done = False
        episode_reward = 0
        episode_trades = 0
        episode_wins = 0
        episode_losses = []

        while not done:
            action = agent.act(state)
            next_state, reward, done, info = env.step(action)
            
            episode_reward += reward
            if info.get('trade_made', False):
                episode_trades += 1
                total_trades += 1
                trade_amount = info.get('trade_amount', 0)
                total_trade_amount += trade_amount
                
                if reward > 0:
                    episode_wins += 1
                    winning_trades += 1
                    weighted_win_sum += trade_amount * (reward / trade_amount)  # Weight by ROI
                
                if action == 1:  # Buy
                    buy_trades += 1
                    if reward > 0:
                        buy_wins += 1
                elif action == 2:  # Sell
                    sell_trades += 1
                    if reward > 0:
                        sell_wins += 1

            loss = agent.calculate_loss(state, action, reward, next_state, done)
            episode_losses.append(loss)
            
            state = next_state

        all_rewards.append(episode_reward)
        if episode_trades > 0:
            all_win_rates.append(episode_wins / episode_trades)
        all_losses.append(np.mean(episode_losses) if episode_losses else 0)

        portfolio_values.append(episode_reward)
        if len(portfolio_values) > 1:
            returns.append((portfolio_values[-1] - portfolio_values[-2]) / portfolio_values[-2])

    buy_win_rate = buy_wins / buy_trades if buy_trades > 0 else 0
    sell_win_rate = sell_wins / sell_trades if sell_trades > 0 else 0
    overall_win_rate = winning_trades / total_trades if total_trades > 0 else 0
    weighted_win_rate = weighted_win_sum / total_trade_amount if total_trade_amount > 0 else 0

    # Calculate additional metrics
    sharpe_ratio = agent.calculate_sharpe_ratio(returns)
    max_drawdown = agent.calculate_max_drawdown(portfolio_values)
    sortino_ratio = agent.calculate_sortino_ratio(returns)
    calmar_ratio = agent.calculate_calmar_ratio(returns, max_drawdown)
    omega_ratio = agent.calculate_omega_ratio(returns)

    return {
        'mean_reward': np.mean(all_rewards),
        'std_reward': np.std(all_rewards),
        'mean_win_rate': np.mean(all_win_rates) if all_win_rates else 0,
        'mean_loss': np.mean(all_losses),
        'total_trades': total_trades,
        'overall_win_rate': overall_win_rate,
        'buy_win_rate': buy_win_rate,
        'sell_win_rate': sell_win_rate,
        'weighted_win_rate': weighted_win_rate,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'sortino_ratio': sortino_ratio,
        'calmar_ratio': calmar_ratio,
        'omega_ratio': omega_ratio
    }

def plot_training_results(rewards, epsilons, win_rates, episode, save_path=TRAINING_RESULTS_PATH):
    plt.figure(figsize=(20, 15))
    
    # Plot rewards over episodes
    plt.subplot(3, 2, 1)
    plt.plot(rewards, label='Rewards', color='b')
    plt.title('Rewards over Episodes')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend()

    # Plot epsilon over episodes
    plt.subplot(3, 2, 2)
    plt.plot(epsilons, label='Epsilon', color='g')
    plt.title('Epsilon over Episodes')
    plt.xlabel('Episode')
    plt.ylabel('Epsilon')
    plt.legend()

    # Plot cumulative rewards
    plt.subplot(3, 2, 3)
    cumulative_rewards = np.cumsum(rewards)
    plt.plot(cumulative_rewards, label='Cumulative Rewards', color='m')
    plt.title('Cumulative Rewards over Episodes')
    plt.xlabel('Episode')
    plt.ylabel('Cumulative Reward')
    plt.legend()

    # Plot reward distribution
    plt.subplot(3, 2, 4)
    plt.hist(rewards, bins=20, color='c', alpha=0.7)
    plt.title('Reward Distribution')
    plt.xlabel('Reward')
    plt.ylabel('Frequency')

    # Plot weighted win rate
    plt.subplot(3, 2, 5)
    plt.plot(win_rates, label='Weighted Win Rate', color='y')
    plt.title('Weighted Win Rate over Episodes')
    plt.xlabel('Episode')
    plt.ylabel('Weighted Win Rate')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f"training_results_{episode}.png"))
    plt.close()

def plot_3d_metrics(rewards, win_rates, losses, episode, save_path=TRAINING_RESULTS_PATH):
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(rewards, win_rates, losses, c=win_rates, cmap='viridis', marker='o')
    ax.set_xlabel('Rewards')
    ax.set_ylabel('Win Rates')
    ax.set_zlabel('Losses')
    ax.set_title('3D Visualization of Training Metrics')
    plt.savefig(os.path.join(save_path, f"3d_metrics_{episode}.png"))
    plt.close()

def explain_model(model, env):
    try:
        # Create a batch of initial states
        initial_states = [env.reset() for _ in range(100)]
        
        # Convert the list of states to a numpy array
        initial_states_array = np.array(initial_states)
        
        # Ensure all elements are numeric
        if not np.issubdtype(initial_states_array.dtype, np.number):
            logger.warning("Non-numeric data detected in states. Converting to float.")
            initial_states_array = initial_states_array.astype(float)
        
        # Convert numpy array to torch tensor
        initial_states_tensor = torch.FloatTensor(initial_states_array).to(model.device)
        
        # Create SHAP explainer
        explainer = shap.DeepExplainer(model, initial_states_tensor)
        
        # Generate SHAP values
        shap_values = explainer.shap_values(initial_states_tensor[0:1])
        
        # Create and save summary plot
        shap.summary_plot(shap_values, initial_states_tensor[0:1], 
                          feature_names=['Open', 'High', 'Low', 'Close', 'Volume', 'Balance', 'Position', 'Total Trades', 'Win Rate', 'Risk-free Asset', 'Volume Ratio'],
                          show=False)
        plt.savefig(os.path.join(TRAINING_RESULTS_PATH, "shap_summary.png"))
        plt.close()
        
        logger.info("SHAP explanation generated and saved successfully.")
    except Exception as e:
        logger.error(f"Error in generating SHAP explanation: {e}")
        logger.info("Skipping SHAP explanation due to error.")

def plot_individual_plots(rewards, epsilons, win_rates, losses, episode, save_path=TRAINING_RESULTS_PATH):
    # Plot rewards over episodes
    plt.figure(figsize=(10, 6))
    plt.plot(rewards, label='Rewards', color='b')
    plt.title('Rewards over Episodes')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend()
    plt.savefig(os.path.join(save_path, f"rewards_{episode}.png"))
    plt.close()

    # Plot epsilon over episodes
    plt.figure(figsize=(10, 6))
    plt.plot(epsilons, label='Epsilon', color='g')
    plt.title('Epsilon over Episodes')
    plt.xlabel('Episode')
    plt.ylabel('Epsilon')
    plt.legend()
    plt.savefig(os.path.join(save_path, f"epsilons_{episode}.png"))
    plt.close()

    # Plot cumulative rewards
    plt.figure(figsize=(10, 6))
    cumulative_rewards = np.cumsum(rewards)
    plt.plot(cumulative_rewards, label='Cumulative Rewards', color='m')
    plt.title('Cumulative Rewards over Episodes')
    plt.xlabel('Episode')
    plt.ylabel('Cumulative Reward')
    plt.legend()
    plt.savefig(os.path.join(save_path, f"cumulative_rewards_{episode}.png"))
    plt.close()

    # Plot reward distribution
    plt.figure(figsize=(10, 6))
    plt.hist(rewards, bins=20, color='c', alpha=0.7)
    plt.title('Reward Distribution')
    plt.xlabel('Reward')
    plt.ylabel('Frequency')
    plt.savefig(os.path.join(save_path, f"reward_distribution_{episode}.png"))
    plt.close()

    # Plot weighted win rate
    plt.figure(figsize=(10, 6))
    plt.plot(win_rates, label='Weighted Win Rate', color='y')
    plt.title('Weighted Win Rate over Episodes')
    plt.xlabel('Episode')
    plt.ylabel('Weighted Win Rate')
    plt.ylim(0, max(1, max(win_rates)))  # Set y-axis limit from 0 to 1 or the max value if it's higher
    plt.legend()
    plt.savefig(os.path.join(save_path, f"win_rates_{episode}.png"))
    plt.close()

    # Plot losses
    plt.figure(figsize=(10, 6))
    plt.plot(losses, label='Loss', color='r')
    plt.title('Loss over Episodes')
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(save_path, f"losses_{episode}.png"))
    plt.close()

def plot_trading_decisions(price_data, actions, episode, buy_signals, sell_signals):
    if not price_data:
        logger.warning("No price data available for plotting trading decisions")
        return

    plt.figure(figsize=(20, 10))
    plt.plot(price_data, label='Price', color='black')
    
    if buy_signals:
        buy_prices = [price_data[min(i, len(price_data)-1)] for i in buy_signals]
        plt.scatter(buy_signals, buy_prices, marker='^', color='g', label='Buy', s=100)
    
    if sell_signals:
        sell_prices = [price_data[min(i, len(price_data)-1)] for i in sell_signals]
        plt.scatter(sell_signals, sell_prices, marker='v', color='r', label='Sell', s=100)
    
    plt.title(f'Trading Decisions Over Time - Episode {episode+1}')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.savefig(os.path.join(TRAINING_RESULTS_PATH, f"trading_decisions_{episode+1}.png"))
    plt.close()

def plot_position_openings(long_positions, short_positions, episode):
    plt.figure(figsize=(10, 6))
    plt.bar(['Long Positions', 'Short Positions'], [long_positions, short_positions], color=['g', 'r'])
    plt.title(f'Position Openings - Episode {episode+1}')
    plt.xlabel('Position Type')
    plt.ylabel('Count')
    plt.savefig(os.path.join(TRAINING_RESULTS_PATH, f"position_openings_{episode+1}.png"))
    plt.close()

def plot_training_sample(train_data, episode):
    plt.figure(figsize=(20, 10))
    plt.plot(train_data['close'], label='Training Data', color='blue')
    plt.title('Training Sample')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.savefig(os.path.join(TRAINING_RESULTS_PATH, f"training_sample_{episode}.png"))
    plt.close()

def plot_testing_sample(test_data, predictions, episode):
    plt.figure(figsize=(20, 10))
    plt.plot(test_data['close'], label='Testing Data', color='orange')
    plt.scatter(range(len(predictions)), predictions, label='Predictions', color='red', marker='x')
    plt.title('Testing Sample with Predictions')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.savefig(os.path.join(TESTING_RESULTS_PATH, f"testing_sample_{episode}.png"))
    plt.close()

def plot_correlation(test_data, predictions, episode):
    plt.figure(figsize=(10, 6))
    plt.scatter(test_data['close'], predictions, label='Correlation', color='purple')
    plt.title('Correlation between Actual and Predicted Prices')
    plt.xlabel('Actual Prices')
    plt.ylabel('Predicted Prices')
    plt.legend()
    plt.savefig(os.path.join(TESTING_RESULTS_PATH, f"correlation_{episode}.png"))
    plt.close()

def save_model_details(agent, episode, score, weighted_win_rate, episode_reward, actions, avg_loss):
    details = {
        "Episode": episode,
        "Score": score,
        "Weighted Win Rate": weighted_win_rate,
        "Total Reward": episode_reward,
        "Epsilon": agent.epsilon,
        "Total Trades": agent.total_trades,
        "Win Rate": agent.calculate_metrics()['win_rate'],
        "Action Distribution": agent.get_action_distribution(),
        "Average Loss": avg_loss,
        "Actions": actions
    }
    
    file_path = os.path.join(TRAINING_RESULTS_PATH, f"model_details_{episode}.txt")
    with open(file_path, 'w') as f:
        for key, value in details.items():
            f.write(f"{key}: {value}\n")

def plot_gradient_descent(agent, episode, save_path=TRAINING_RESULTS_PATH):
    plt.figure(figsize=(10, 6))
    plt.plot(agent.gradient_norms, label='Gradient Norm')
    plt.title('Gradient Norm over Training Steps')
    plt.xlabel('Training Step')
    plt.ylabel('Gradient Norm')
    plt.legend()
    plt.savefig(os.path.join(save_path, f"gradient_descent_{episode}.png"))
    plt.close()

def analyze_shap_values(shap_values, expected_value, feature_names, episode):
    try:
        shap_sum = np.abs(shap_values[0]).mean(axis=0)
        importance_df = pd.DataFrame(list(zip(feature_names, shap_sum)), columns=['feature', 'importance'])
        importance_df = importance_df.sort_values('importance', ascending=False)

        analysis_text = f"SHAP Analysis for Episode {episode}:\n\n"
        analysis_text += f"Expected value: {expected_value[0]:.4f}\n\n"
        analysis_text += "Top 5 most important features:\n"
        for _, row in importance_df.head().iterrows():
            analysis_text += f"{row['feature']}: {row['importance']:.4f}\n"

        analysis_text += "\nFeature impact analysis:\n"
        for _, row in importance_df.iterrows():
            feature = row['feature']
            importance = row['importance']
            mean_shap = np.mean(shap_values[0][:, feature_names.index(feature)])
            if mean_shap > 0:
                impact = "positive"
            elif mean_shap < 0:
                impact = "negative"
            else:
                impact = "neutral"
            analysis_text += f"{feature}: {importance:.4f} ({impact} impact)\n"

        with open(os.path.join(TRAINING_RESULTS_PATH, f"shap_analysis_{episode}.txt"), 'w') as f:
            f.write(analysis_text)

        logger.info(f"SHAP analysis saved for episode {episode}")
    except Exception as e:
        logger.error(f"Error in analyzing SHAP values: {e}")

if __name__ == "__main__":
    try:
        # Get all JSON files in the current directory
        json_files = [f for f in os.listdir('.') if f.endswith('.json')]
        
        if len(json_files) < 3:
            logger.error("Not enough JSON files found. Need at least 3 files.")
            sys.exit(1)
        
        # Randomly select 2 files for testing
        test_files = random.sample(json_files, 2)
        train_files = [f for f in json_files if f not in test_files]

        logger.info(f"Training files: {train_files}")
        logger.info(f"Testing files: {test_files}")

        # Load and process training data
        train_data = pd.concat([load_json_data(file) for file in train_files])
        logger.info(f"Loaded {len(train_data)} total records for training")

        # Create training environment
        train_env = TradingEnvironment(train_data.to_dict('records'), trade_fraction=0.2)  # Add trade_fraction parameter

        # Initialize and train agent
        state_size = train_env._get_state().shape[0]
        action_size = 4  # hold, buy, sell, risk-free
        agent = ImprovedQLearningAgent(
            state_size=state_size,
            action_size=action_size,
            learning_rate=0.0001,
            batch_size=32,
            epsilon_start=0.5,  # Start epsilon
            epsilon_end=0.05,   # End epsilon
            epsilon_decay_steps=130,
            feature_names=['Open', 'High', 'Low', 'Close', 'Volume', 'SMA20', 'SMA50', 'RSI', 'MACD', 'Signal', 'Upper BB', 'Lower BB', 'Balance', 'Position', 'Total Trades', 'Win Rate', 'Risk-free Asset', 'Volume Ratio']
        )
        trained_agent, all_rewards, all_epsilons, all_win_rates, all_losses = train_agent(
            train_env, agent, episodes=130, batch_size=64
        )

        # Plot final training results
        plot_training_results(all_rewards, all_epsilons, all_win_rates, 'final')
        plot_3d_metrics(all_rewards, all_win_rates, all_losses, 'final')
        plot_individual_plots(all_rewards, all_epsilons, all_win_rates, all_losses, 'final')
        plot_action_distribution(trained_agent, 'final', TRAINING_RESULTS_PATH)

        # Plot training sample
        plot_training_sample(train_data, 'final')

        # Explain model using SHAP
        try:
            sample_data = train_data.sample(n=100).select_dtypes(include=[np.number]).to_numpy()
            feature_names = train_data.select_dtypes(include=[np.number]).columns.tolist()
            plot_shap_values(trained_agent.model, sample_data, feature_names, TRAINING_RESULTS_PATH, 'final')
        except Exception as e:
            logger.error(f"Error in SHAP explanation: {e}")
            logger.info("Skipping SHAP explanation due to error.")

        # Evaluate agent on test data
        for test_file in test_files:
            test_data = load_json_data(test_file)
            logger.info(f"Evaluating on {test_file} with {len(test_data)} records")

            test_env = TradingEnvironment(test_data.to_dict('records'))
            test_metrics = evaluate_agent(test_env, trained_agent)
            
            logger.info(f"Test Metrics for {test_file}:")
            logger.info(f"  Mean Reward: {test_metrics['mean_reward']:.2f} ± {test_metrics['std_reward']:.2f}")
            logger.info(f"  Mean Win Rate: {test_metrics['mean_win_rate']:.4f}")
            logger.info(f"  Overall Win Rate: {test_metrics['overall_win_rate']:.4f}")
            logger.info(f"  Buy Win Rate: {test_metrics['buy_win_rate']:.4f}")
            logger.info(f"  Sell Win Rate: {test_metrics['sell_win_rate']:.4f}")
            logger.info(f"  Weighted Win Rate: {test_metrics['weighted_win_rate']:.4f}")
            logger.info(f"  Mean Loss: {test_metrics['mean_loss']:.4f}")
            logger.info(f"  Total Trades: {test_metrics['total_trades']}")
            logger.info(f"  Sharpe Ratio: {test_metrics['sharpe_ratio']:.4f}")
            logger.info(f"  Max Drawdown: {test_metrics['max_drawdown']:.4f}")
            logger.info(f"  Sortino Ratio: {test_metrics['sortino_ratio']:.4f}")
            logger.info(f"  Calmar Ratio: {test_metrics['calmar_ratio']:.4f}")
            logger.info(f"  Omega Ratio: {test_metrics['omega_ratio']:.4f}")

            # Generate predictions
            predictions = []
            state = test_env.reset()
            done = False
            while not done:
                action = trained_agent.act(state)
                next_state, _, done, _ = test_env.step(action)
                predictions.append(test_env.data[test_env.current_step]['close'])
                state = next_state

            # Plot testing sample with predictions
            plot_testing_sample(test_data, predictions, f'final_{test_file}')

            # Plot correlation
            plot_correlation(test_data, predictions, f'final_{test_file}')

            # Calculate additional metrics
            sharpe_ratio = calculate_sharpe_ratio(test_metrics['mean_reward'], test_metrics['std_reward'])
            max_drawdown = calculate_max_drawdown(predictions)
            profit_factor = calculate_profit_factor(predictions)

            logger.info(f"  Sharpe Ratio: {sharpe_ratio:.4f}")
            logger.info(f"  Max Drawdown: {max_drawdown:.4f}")
            logger.info(f"  Profit Factor: {profit_factor:.4f}")

            # Save test metrics to a file
            test_metrics_file = os.path.join(TESTING_RESULTS_PATH, f"test_metrics_{test_file}.txt")
            with open(test_metrics_file, 'w') as f:
                for key, value in test_metrics.items():
                    f.write(f"{key}: {value}\n")
                f.write(f"Sharpe Ratio: {sharpe_ratio:.4f}\n")
                f.write(f"Max Drawdown: {max_drawdown:.4f}\n")
                f.write(f"Profit Factor: {profit_factor:.4f}\n")

        # Save final model
        trained_agent.save(os.path.join(TRAINING_RESULTS_PATH, "final_model.pth"))

        # Log final metrics
        final_metrics = trained_agent.calculate_metrics()
        logger.info("Final metrics:")
        logger.info(f"Total trades: {final_metrics['total_trades']}")
        logger.info(f"Win rate: {final_metrics['win_rate']:.2f}")

    except Exception as e:
        logger.error(f"An error occurred: {e}")
        sys.exit(1)

