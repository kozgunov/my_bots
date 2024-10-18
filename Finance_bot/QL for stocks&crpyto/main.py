from environment import TradingEnvironment
from agent import ImprovedQLearningAgent
from training import train_agent, test_agent
from utils import load_real_data, load_and_prepare_data, get_sentiment_scores
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from xgboost import XGBRegressor
from fbprophet import Prophet
from bayes_opt import BayesianOptimization
from sklearn.metrics import mean_squared_error
from sklearn.feature_selection import SelectKBest, f_regression
import matplotlib.pyplot as plt

def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:(i + seq_length)]
        y = data[i + seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

def build_lstm_model(input_shape):
    model = Sequential([
        LSTM(50, activation='relu', input_shape=input_shape, return_sequences=True),
        Dropout(0.2),
        LSTM(50, activation='relu', return_sequences=False),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    return model

def optimize_xgboost(X, y):
    def xgb_evaluate(max_depth, learning_rate, n_estimators, min_child_weight, subsample, colsample_bytree):
        params = {
            'max_depth': int(max_depth),
            'learning_rate': learning_rate,
            'n_estimators': int(n_estimators),
            'min_child_weight': min_child_weight,
            'subsample': subsample,
            'colsample_bytree': colsample_bytree
        }
        model = XGBRegressor(**params)
        model.fit(X, y)
        predictions = model.predict(X)
        return -mean_squared_error(y, predictions)

    pbounds = {
        'max_depth': (3, 10),
        'learning_rate': (0.01, 0.3),
        'n_estimators': (100, 1000),
        'min_child_weight': (1, 10),
        'subsample': (0.5, 1.0),
        'colsample_bytree': (0.5, 1.0)
    }

    optimizer = BayesianOptimization(f=xgb_evaluate, pbounds=pbounds, random_state=1)
    optimizer.maximize(init_points=5, n_iter=25)

    return optimizer.max

def main():
    # Load and prepare hourly data
    df, scaler = load_and_prepare_data('BTC-USD', period="60d", interval="1h")
    
    if df is None:
        print("Failed to load data. Exiting.")
        return

    # Add sentiment scores (you might need to adjust this for hourly data)
    df['Sentiment'] = get_sentiment_scores('BTC-USD', df.index)

    # Prepare features
    features = ['Close', 'SMA_20', 'SMA_50', 'RSI', 'MACD', 'MACD_Signal', 'Returns', 'Sentiment']
    X = df[features].values
    y = df['Close'].values

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create environment and agent
    env = TradingEnvironment(y_test)
    agent = ImprovedQLearningAgent(state_size=4, action_size=3)  # 3 actions: hold, buy, sell

    # Training
    episodes = 1000
    batch_size = 32
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            action = agent.get_action(np.array([state]))
            next_state, reward, done = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            if len(agent.memory) > batch_size:
                agent.replay(batch_size)

        if episode % 10 == 0:
            print(f"Episode: {episode}, Total Reward: {total_reward}, Epsilon: {agent.epsilon:.2f}")

    # Testing
    state = env.reset()
    total_reward = 0
    done = False
    actions = []

    while not done:
        action = agent.get_action(np.array([state]))
        next_state, reward, done = env.step(action)
        state = next_state
        total_reward += reward
        actions.append(action)

    print(f"Final balance: {env.balance:.2f}")
    print(f"Total reward: {total_reward:.2f}")

    # Plotting results
    plt.figure(figsize=(12, 6))
    plt.plot(y_test, label='Bitcoin Price')
    buy_points = [i for i, a in enumerate(actions) if a == 1]
    sell_points = [i for i, a in enumerate(actions) if a == 2]
    plt.scatter(buy_points, y_test[buy_points], color='green', label='Buy', marker='^')
    plt.scatter(sell_points, y_test[sell_points], color='red', label='Sell', marker='v')
    plt.xlabel('Hours')
    plt.ylabel('Price')
    plt.title('Bitcoin Trading Bot Actions')
    plt.legend()
    plt.savefig('trading_results.png')
    plt.close()

if __name__ == "__main__":
    main()
