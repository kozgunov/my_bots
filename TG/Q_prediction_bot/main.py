from environment import TradingEnvironment
from agent import ImprovedQLearningAgent
from utils import prepare_data, create_sequences
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from telegram_bot import send_notification, send_confirmation, start_bot
import threading
from config import *

def main():
    # Start the Telegram bot in a separate thread
    bot_thread = threading.Thread(target=start_bot)
    bot_thread.start()

    # Load and prepare hourly data for Bitcoin
    df, scaler = prepare_data('binance', SYMBOL, timeframe=TIMEFRAME, limit=DATA_LIMIT)
    
    if df is None:
        print("Failed to load data. Exiting.")
        return

    # Prepare features and target
    features = df.columns.drop('close')
    target = 'close'
    
    # Create sequences
    seq_length = 24  # 24 hours of data
    X, y = create_sequences(df, seq_length)
    y = y[:, df.columns.get_loc(target)]  # We only want to predict the 'close' price

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create environment and agent
    env = TradingEnvironment(y_test, initial_balance=INITIAL_BALANCE, fee=TRADING_FEE)
    agent = ImprovedQLearningAgent(state_size=X.shape[1] * X.shape[2] + 2, action_size=3, 
                                   learning_rate=LEARNING_RATE, gamma=GAMMA, epsilon=EPSILON, 
                                   epsilon_decay=EPSILON_DECAY, epsilon_min=EPSILON_MIN, 
                                   memory_size=MEMORY_SIZE, batch_size=BATCH_SIZE)

    # Training
    for episode in range(EPISODES):
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            action = agent.get_action(np.array([state]))
            next_state, reward, done = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            if len(agent.memory) > BATCH_SIZE:
                agent.replay()

        if episode % 10 == 0:
            print(f"Episode: {episode}, Total Reward: {total_reward}, Epsilon: {agent.epsilon:.2f}")

    # Testing with Telegram notifications and confirmations
    state = env.reset()
    total_reward = 0
    done = False
    actions = []
    balances = []
    holdings = []

    while not done:
        action = agent.get_action(np.array([state]))
        current_price = env.data[env.current_step]
        
        if action == 1:  # Buy
            confirmation = send_confirmation("buy", current_price)
            if confirmation:
                next_state, reward, done = env.step(action)
                send_notification(f"Bought at price {current_price}")
            else:
                next_state, reward, done = env.step(0)  # Hold instead
        elif action == 2:  # Sell
            confirmation = send_confirmation("sell", current_price)
            if confirmation:
                next_state, reward, done = env.step(action)
                send_notification(f"Sold at price {current_price}")
            else:
                next_state, reward, done = env.step(0)  # Hold instead
        else:  # Hold
            next_state, reward, done = env.step(action)
        
        state = next_state
        total_reward += reward
        actions.append(action)
        balances.append(env.balance)
        holdings.append(env.holdings)

    print(f"Final balance: {env.balance:.2f}")
    print(f"Final holdings: {env.holdings:.2f}")
    print(f"Total reward: {total_reward:.2f}")

    # Plotting results
    plt.figure(figsize=(12, 8))
    plt.subplot(3, 1, 1)
    plt.plot(y_test, label='Price')
    plt.title('Price, Actions, and Portfolio Value')
    plt.legend()

    plt.subplot(3, 1, 2)
    plt.plot(actions, label='Actions')
    plt.ylabel('Action (0: Hold, 1: Buy, 2: Sell)')
    plt.legend()

    plt.subplot(3, 1, 3)
    portfolio_values = np.array(balances) + np.array(holdings) * y_test
    plt.plot(portfolio_values, label='Portfolio Value')
    plt.xlabel('Time Step')
    plt.legend()

    plt.tight_layout()
    plt.savefig('trading_results.png')
    plt.close()

    # Send final notification
    send_notification(f"Trading session completed.\nFinal balance: {env.balance:.2f}\nFinal holdings: {env.holdings:.2f}\nTotal reward: {total_reward:.2f}")

if __name__ == "__main__":
    main()
