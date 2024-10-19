import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
import tensorflow as tf

def train_agent(env, agent, episodes, batch_size=32, update_target_every=5, checkpoint_every=100, convergence_threshold=0.01):
    rewards_history = []
    avg_rewards_history = []
    epsilon_history = []
    best_avg_reward = -np.inf
    convergence_counter = 0

    checkpoint_dir = './checkpoints'
    checkpoint = tf.train.Checkpoint(model=agent.model)
    manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=3)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    for fold, (train_index, val_index) in enumerate(kf.split(range(episodes))):
        print(f"Starting fold {fold + 1}")
        
        for episode in range(episodes // 5):
            state = env.reset()
            state = np.reshape(state, [1, 3])
            total_reward = 0
            done = False

            while not done:
                action = agent.get_action(state)
                next_state, reward, done = env.step(action)
                next_state = np.reshape(next_state, [1, 3])
                agent.remember(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward

                agent.replay()
                agent.delta_epsilon_update(reward)

            if episode % update_target_every == 0:
                agent.update_target_model()

            rewards_history.append(total_reward)
            avg_reward = np.mean(rewards_history[-100:])
            avg_rewards_history.append(avg_reward)
            epsilon_history.append(agent.epsilon)

            if episode % checkpoint_every == 0:
                manager.save()
                print(f"Checkpoint saved at episode {episode}")

            if avg_reward > best_avg_reward:
                best_avg_reward = avg_reward
                agent.save("best_model.h5")
                convergence_counter = 0
            else:
                convergence_counter += 1

            if convergence_counter >= 100 and avg_reward > convergence_threshold:
                print(f"Converged at episode {episode}")
                break

            if episode % 100 == 0:
                print(f"Fold {fold + 1}, Episode: {episode}, Total Reward: {total_reward:.2f}, Avg Reward: {avg_reward:.2f}, Epsilon: {agent.epsilon:.2f}")

    plot_training_results(rewards_history, avg_rewards_history, epsilon_history)

def test_agent(env, agent):
    state = env.reset()
    state = np.reshape(state, [1, 3])
    total_reward = 0
    done = False
    actions_history = []

    while not done:
        action = agent.get_action(state)
        next_state, reward, done = env.step(action)
        next_state = np.reshape(next_state, [1, 3])
        state = next_state
        total_reward += reward
        actions_history.append(action)

    plot_test_results(env.data, actions_history)
    return env.balance, env.holdings, total_reward

def plot_training_results(rewards, avg_rewards, epsilons):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))

    ax1.plot(rewards, label='Reward')
    ax1.plot(avg_rewards, label='100-episode Average')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')
    ax1.set_title('Training Rewards')
    ax1.legend()

    ax2.plot(epsilons)
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Epsilon')
    ax2.set_title('Epsilon Decay')

    plt.tight_layout()
    plt.savefig('training_results.png')
    plt.close()

def plot_test_results(prices, actions):
    plt.figure(figsize=(12, 6))
    plt.plot(prices, label='Price')
    buy_points = [i for i, a in enumerate(actions) if a == 1]
    sell_points = [i for i, a in enumerate(actions) if a == 2]
    plt.scatter(buy_points, [prices[i] for i in buy_points], color='green', label='Buy', marker='^')
    plt.scatter(sell_points, [prices[i] for i in sell_points], color='red', label='Sell', marker='v')
    plt.xlabel('Time Step')
    plt.ylabel('Price')
    plt.title('Agent Actions on Test Data')
    plt.legend()
    plt.savefig('test_results.png')
    plt.close()
