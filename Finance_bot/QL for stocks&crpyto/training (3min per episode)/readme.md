# Cryptocurrency Trading Bot with Deep Q-Learning

## Overview

This project implements an advanced cryptocurrency trading bot using Deep Q-Learning, a reinforcement learning technique. The bot is designed to make trading decisions in a simulated cryptocurrency market environment, aiming to maximize profits while managing risk.

## Key Components

1. **Environment (environment.py)**: Simulates the cryptocurrency market.
2. **Agent (agent.py)**: Implements the Deep Q-Learning algorithm.
3. **Training Process (training.py)**: Manages the training of the agent.
4. **Utilities (utils.py)**: Provides helper functions and data processing tools.

## Detailed Process

### 1. Data Preparation

- Historical cryptocurrency data is loaded and preprocessed.
- Technical indicators are calculated, including:
  - Simple Moving Averages (SMA)
  - Relative Strength Index (RSI)
  - Moving Average Convergence Divergence (MACD)
  - Bollinger Bands
  - Average True Range (ATR)
  - Commodity Channel Index (CCI)

### 2. Environment Setup

- The `TradingEnvironment` class simulates the market:
  - Manages account balance, positions, and trades
  - Implements realistic features like trading fees and price impact
  - Provides market state information to the agent

### 3. Agent Architecture

- Uses a Deep Q-Network (DQN) with the following structure:
  - Input layer: State size (market features)
  - Hidden layers: 256 -> 128 -> 64 -> 32 neurons
  - Output layer: 4 neurons (hold, buy, sell, move to risk-free asset)
- Implements experience replay with prioritized sampling
- Uses epsilon-greedy exploration strategy

### 4. Training Process

- Episodes are run where the agent interacts with the environment
- For each step:
  - Agent selects an action based on the current state
  - Environment executes the action and returns the next state and reward
  - Experience is stored in the replay buffer
- Periodic training of the neural network occurs using sampled experiences
- Gradient norms are tracked to monitor training stability

### 5. Evaluation Metrics

The agent's performance is evaluated using various financial metrics:

| Metric | Value |
|--------|-------|
| Sharpe Ratio | [Placeholder] |
| Max Drawdown | [Placeholder] |
| Profit Factor | [Placeholder] |
| Sortino Ratio | [Placeholder] |
| Calmar Ratio | [Placeholder] |
| Omega Ratio | [Placeholder] |

### 6. SHAP Analysis

SHAP (SHapley Additive exPlanations) values are used to interpret the model's decisions:
- Explains which features are most important for each prediction
- Provides insights into the model's decision-making process

## Visualizations

The training process generates various plots to help understand the agent's performance:

1. **Training Results**:
   [Placeholder for Training Results Plot]

2. **3D Metrics**:
   [Placeholder for 3D Metrics Plot]

3. **Individual Plots**:
   - Rewards over Episodes:
     [Placeholder for Rewards Plot]
   - Epsilon over Episodes:
     [Placeholder for Epsilon Plot]
   - Cumulative Rewards:
     [Placeholder for Cumulative Rewards Plot]
   - Reward Distribution:
     [Placeholder for Reward Distribution Plot]
   - Weighted Win Rate:
     [Placeholder for Weighted Win Rate Plot]
   - Losses:
     [Placeholder for Losses Plot]

4. **Action Distribution**:
   [Placeholder for Action Distribution Plot]

5. **Trading Decisions**:
   [Placeholder for Trading Decisions Plot]

6. **Position Openings**:
   [Placeholder for Position Openings Plot]

7. **SHAP Plots**: 
   - Summary plot:
     [Placeholder for SHAP Summary Plot]
   - Bar plot:
     [Placeholder for SHAP Bar Plot]
   - Force plot:
     [Placeholder for SHAP Force Plot]

8. **Gradient Descent**:
   [Placeholder for Gradient Descent Plot]

## Future Improvements

- Implement more advanced RL algorithms (e.g., PPO, SAC)
- Incorporate sentiment analysis from news and social media
- Expand to multi-asset trading
- Implement real-time data streaming for live trading
- Conduct more extensive hyperparameter tuning

## Conclusion

This cryptocurrency trading bot demonstrates the application of deep reinforcement learning in financial markets. While showing promising results in simulations, it's important to note that real-world trading involves additional complexities and risks not fully captured in this model.
