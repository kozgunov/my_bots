# (without generating data)

import numpy as np
import matplotlib.pyplot as plt
from environment import TradingEnvironment
from agent import ImprovedQLearningAgent
from utils import prepare_data, create_sequences
from config import *
from sklearn.model_selection import train_test_split
import torch
import os
import pandas as pd
import seaborn as sns
import logging
import shap
import json
import time

# setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DESKTOP_PATH = os.path.join(os.path.expanduser("~"), "Desktop")
RESULTS_PATH = os.path.join(DESKTOP_PATH, "training_results")

os.makedirs(RESULTS_PATH, exist_ok=True)

def load_json_data(file_path, chunk_size=100000):
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    if not isinstance(data, list):
        raise ValueError("JSON file should contain a list of objects")
    
    for i in range(0, len(data), chunk_size):
        yield pd.DataFrame(data[i:i+chunk_size])

def train_agent(env, agent, episodes, batch_size=64, checkpoint_interval=10, patience=50):
    best_score = -np.inf
    best_loss = float('inf')
    scores = []
    steps_per_episode = []
    weighted_win_rates = []
    losses = []
    epsilons = []
    no_improvement_count = 0

    logger.info(f"Starting training for {episodes} episodes")

    for episode in range(episodes):
        start_time = time.time()
        state = env.reset()
        done = False
        episode_reward = 0
        episode_steps = 0
        episode_trades = 0
        episode_wins = 0
        episode_losses = []

        while not done and time.time() - start_time < 60:  # 1 minute time limit
            action = agent.act(state)
            next_state, reward, done, info = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            episode_reward += reward
            episode_steps += 1

            if info['total_trades'] > episode_trades:
                episode_trades = info['total_trades']
                if reward > 0:
                    episode_wins += 1

            loss = agent.replay(episode * 10000 + episode_steps)
            if loss is not None:
                episode_losses.append(loss)

        #  episode metrics
        weighted_win_rate = episode_wins / episode_trades if episode_trades > 0 else 0
        scores.append(episode_reward)
        steps_per_episode.append(episode_steps)
        weighted_win_rates.append(weighted_win_rate)
        epsilons.append(agent.epsilon)
        avg_loss = np.mean(episode_losses) if episode_losses else float('inf')
        losses.append(avg_loss)

        # log episode summary
        logger.info(f"Episode {episode+1}/{episodes} - Time: {time.time() - start_time:.2f}s, "
                    f"Reward: {episode_reward:.2f}, Steps: {episode_steps}, "
                    f"Trades: {episode_trades}, Win Rate: {weighted_win_rate:.2f}, "
                    f"Avg Loss: {avg_loss:.6f}, Epsilon: {agent.epsilon:.4f}")

        # check for improvement and save best model
        if episode_reward > best_score:
            best_score = episode_reward
            agent.save(os.path.join(RESULTS_PATH, f"model_best_score_{episode}.pth"))
            logger.info(f"New best model (score) saved at episode {episode+1}")
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            no_improvement_count = 0
            agent.save(os.path.join(RESULTS_PATH, f"model_best_loss_{episode}.pth"))
            logger.info(f"New best model (loss) saved at episode {episode+1}")
        else:
            no_improvement_count += 1

        # save checkpoint and plot results
        if episode % checkpoint_interval == 0 or episode == episodes - 1:
            agent.save(os.path.join(RESULTS_PATH, f"model_checkpoint_{episode}.pth"))
            plot_training_results(scores, steps_per_episode, weighted_win_rates, losses, epsilons, episode)
            logger.info(f"Checkpoint saved and plots generated at episode {episode+1}")

        # early stopping
        if no_improvement_count >= patience:
            logger.info(f"Stopping early due to no improvement in loss for {patience} episodes")
            break

    # final summary
    logger.info("Training completed")
    logger.info(f"Best score: {best_score:.2f}")
    logger.info(f"Best loss: {best_loss:.6f}")

    # save the final model
    agent.save(os.path.join(RESULTS_PATH, f"model_final.pth"))
    logger.info("Final model saved")

    # generate final plots
    plot_training_results(scores, steps_per_episode, weighted_win_rates, losses, epsilons, 'final')

    return agent

def evaluate_agent(env, agent, episodes=100):
    scores = []
    for episode in range(episodes):
        state = env.reset()
        done = False
        score = 0
        while not done:
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            state = next_state
            score += reward
        scores.append(score)
    return np.mean(scores), np.std(scores)

def plot_training_results(scores, steps, win_rates, losses, epsilons, episode):
    #  scores
    plt.figure(figsize=(10, 5))
    plt.plot(scores)
    plt.title('Score over episodes')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.savefig(os.path.join(RESULTS_PATH, f"scores_episode_{episode}.png"))
    plt.close()

    #  steps per episode
    plt.figure(figsize=(10, 5))
    plt.plot(steps)
    plt.title('Steps per episode')
    plt.xlabel('Episode')
    plt.ylabel('Steps')
    plt.savefig(os.path.join(RESULTS_PATH, f"steps_episode_{episode}.png"))
    plt.close()

    #  weighted win rates
    plt.figure(figsize=(10, 5))
    plt.plot(win_rates)
    plt.title('Weighted Win Rate')
    plt.xlabel('Episode')
    plt.ylabel('Win Rate')
    plt.savefig(os.path.join(RESULTS_PATH, f"win_rates_episode_{episode}.png"))
    plt.close()

    #  losses
    if losses:
        plt.figure(figsize=(10, 5))
        plt.plot(losses)
        plt.title('Loss over training steps')
        plt.xlabel('Training step')
        plt.ylabel('Loss')
        plt.savefig(os.path.join(RESULTS_PATH, f"losses_episode_{episode}.png"))
        plt.close()

    #  epsilon decay
    plt.figure(figsize=(10, 5))
    plt.plot(epsilons)
    plt.title('Epsilon Decay')
    plt.xlabel('Episode')
    plt.ylabel('Epsilon')
    plt.savefig(os.path.join(RESULTS_PATH, f"epsilon_decay_episode_{episode}.png"))
    plt.close()

def explain_model(model, env):
    try:
        # create a batch of initial states
        initial_states = [env.reset() for _ in range(100)]
        
        # convert the list of states to a numpy array
        initial_states_array = np.array(initial_states)
        
        # convert numpy array to torch tensor
        initial_states_tensor = torch.FloatTensor(initial_states_array).to(model.device)
        
        # create SHAP explainer
        explainer = shap.DeepExplainer(model, initial_states_tensor)
        
        # generate SHAP values
        shap_values = explainer.shap_values(initial_states_tensor[0:1])
        
        # create and save summary plot
        shap.summary_plot(shap_values, initial_states_tensor[0:1], 
                          feature_names=['Balance', 'Position', 'Price', 'Volume', 'Profit/Loss'],
                          show=False)
        plt.savefig(os.path.join(RESULTS_PATH, "shap_summary.png"))
        plt.close()
        
        logger.info("SHAP explanation generated and saved successfully.")
    except Exception as e:
        logger.error(f"Error in generating SHAP explanation: {e}")
        logger.info("Skipping SHAP explanation due to error.")

def load_all_coins_data(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return pd.DataFrame(data)

if __name__ == "__main__":
    all_coins_file = os.path.join(DESKTOP_PATH, "all_coins_data.json")
    
    if not os.path.exists(all_coins_file):
        logger.error(f"File not found: {all_coins_file}")
        exit(1)

    # load all data at once
    all_data = load_all_coins_data(all_coins_file)
    logger.info(f"Loaded {len(all_data)} total records for all coins")

    for coin in ['BTC', 'ETH', 'BNB', 'ADA', 'XRP', 'DOT', 'UNI', 'LTC', 'LINK', 'SOL']:
        logger.info(f"Processing data for {coin}")
        
        # filter data for the current coin
        coin_data = all_data[all_data['coin'] == coin]
        
        if len(coin_data) == 0:
            logger.warning(f"No data found for {coin}. Skipping...")
            continue

        logger.info(f"Processing {len(coin_data)} records for {coin}")

        # sort data by timestamp
        coin_data = coin_data.sort_values('timestamp')

        # split data
        train_data, temp_data = train_test_split(coin_data, test_size=0.4, shuffle=False)
        val_data, test_data = train_test_split(temp_data, test_size=0.5, shuffle=False)

        # create environments
        train_env = TradingEnvironment(train_data.to_dict('records'))
        val_env = TradingEnvironment(val_data.to_dict('records'))
        test_env = TradingEnvironment(test_data.to_dict('records'))

        # initialize and train agent
        state_size = train_env._get_state().shape[0]
        action_size = 3  # hold, buy, sell
        agent = ImprovedQLearningAgent(state_size, action_size)
        trained_agent = train_agent(train_env, agent, episodes=200, batch_size=64)

        # evaluate agent
        val_score, val_std = evaluate_agent(val_env, trained_agent)
        logger.info(f"Validation Score for {coin}: {val_score:.2f} ± {val_std:.2f}")

        test_score, test_std = evaluate_agent(test_env, trained_agent)
        logger.info(f"Test Score for {coin}: {test_score:.2f} ± {test_std:.2f}")

        # save model
        trained_agent.save(os.path.join(RESULTS_PATH, f"model_{coin}.pth"))

        # explain model
        explain_model(trained_agent.model, test_env)

    logger.info("Training and evaluation completed for all coins.")
