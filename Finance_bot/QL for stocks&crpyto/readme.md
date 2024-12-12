# Crypto Trading Bot with Deep Q-Learning (In progress...)

This project implements a cryptocurrency trading bot using Deep Q-Learning with PyTorch. The bot is designed to make trading decisions on 10-minute cryptocurrency data, considering technical indicators and market trends. It also includes a Telegram bot interface for user interaction and real-time notifications.

## Features

- Deep Q-Network (DQN) implementation using PyTorch for trading decisions
- Support for multiple cryptocurrency exchanges (Binance, OKX, Bybit)
- Technical indicator calculation and analysis
- Customizable trading environment with fees and time constraints
- Data preprocessing and normalization
- Visualization of training progress and trading results
- Telegram bot integration for notifications, trading confirmations, and user interactions
- Real-time market data fetching and analysis
- News sentiment analysis for informed decision making
- User data management with PostgreSQL database
- GPU acceleration support for faster training
- Improved model architecture with batch normalization and gradient clipping
- Adaptive learning rate using ReduceLROnPlateau
- Early stopping to prevent overfitting
- Comprehensive logging and visualization of training metrics

## Project Structure

- `main.py`: The main script to run the trading bot and Telegram interface
- `agent.py`: Implementation of the DQN agent using PyTorch
- `environment.py`: Trading environment simulation
- `utils.py`: Utility functions for data loading, preprocessing, and API interactions
- `training.py`: Functions for training and testing the agent
- `config.py`: Configuration settings for the project
- `telegram_bot.py`: Telegram bot implementation for user interactions
- `Q-learning.py`: Basic Q-learning implementation (for reference)

## Requirements

- Python 3.7+
- PyTorch (with CUDA support for GPU acceleration)
- NumPy
- Pandas
- Matplotlib
- scikit-learn
- ccxt
- python-telegram-bot
- psycopg2
- requests
- seaborn

## Installation

1. Clone this repository:   ```
   git clone https://github.com/yourusername/crypto-trading-bot.git
   cd crypto-trading-bot   ```

2. Install the required packages:   ```
   pip install -r requirements.txt   ```

3. Set up your PostgreSQL database and update the connection details in `config.py`.

4. Update the API keys and tokens in `config.py` for the services you plan to use.

## Usage

1. Configure your desired cryptocurrency and parameters in `config.py`
2. Run the training script:   ```
   python training.py   ```
3. After training, run the main script to start the bot:   ```
   python main.py   ```

4. The script will start the Telegram bot and begin monitoring the market. Users can interact with the bot using the provided commands.

## Customization

- Adjust the `ImprovedAgent` parameters in `agent.py` to modify the learning process
- Customize the trading environment in `environment.py` to add more complex trading rules
- Add or remove technical indicators in the `add_technical_indicators` function in `utils.py`
- Modify the Telegram bot commands and responses in `main.py` to suit your needs

## Results

The bot generates performance plots and sends trading notifications through the Telegram interface. You can also view the training progress and trading results in the console output and in the `results` folder on your desktop.

## GPU Acceleration

This project supports GPU acceleration for faster training. Make sure you have CUDA installed and configured properly. The script will automatically use GPU if available.

## Disclaimer

This project is for educational purposes only. Cryptocurrency trading carries a high level of risk, and this bot should not be used for actual trading without thorough testing and risk assessment.

## Contributing

Contributions, issues, and feature requests are welcome. Feel free to check [issues page](https://github.com/yourusername/crypto-trading-bot/issues) if you want to contribute.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.





# Cryptocurrency Trading Bot with Q-Learning

## Overview
This project implements a cryptocurrency trading bot using reinforcement learning techniques, specifically Q-learning with a Deep Q-Network (DQN). The bot is trained on historical cryptocurrency data and can make trading decisions (buy, sell, or hold) based on market conditions.

## Key Components

1. Data Collection (`database_parser.py`)
   - Fetches historical data for 10 popular cryptocurrencies from the Binance API
   - Processes and saves data in JSON format
   - Stores data for each coin individually and combined in the "training_results" folder

2. Trading Environment (`environment.py`)
   - Simulates a trading environment with account balance, positions, and trades
   - Implements trading logic including fees and risk management
   - Provides state representation and calculates rewards

3. Reinforcement Learning Agent (`agent.py`)
   - Implements an Improved Q-Learning Agent with a Dueling DQN architecture
   - Uses Prioritized Experience Replay for efficient learning
   - Implements epsilon-greedy exploration strategy

4. Training Process (`training.py`)
   - Manages the overall training loop for each cryptocurrency
   - Implements early stopping and model checkpointing
   - Generates performance plots and logs training progress

## Training Process in Detail

1. Data Preparation
   - Loads all cryptocurrency data from the "all_coins_data.json" file
   - Splits data into training (60%), validation (20%), and testing (20%) sets for each coin
   - Sorts data chronologically to maintain time series integrity

2. Environment and Agent Initialization
   - Creates separate TradingEnvironment instances for training, validation, and testing
   - Initializes the ImprovedQLearningAgent with state and action sizes based on the environment

3. Training Loop
   - Runs for a maximum of 200 episodes or until early stopping criteria are met
   - Each episode is limited to 1 minute of training time

4. Episode Structure
   - Resets the environment at the start of each episode
   - Agent takes actions based on the current state
   - Environment returns next state, reward, and done flag
   - Experiences are stored in the agent's replay memory

5. Learning Process
   - Agent performs experience replay at each step
   - Updates Q-values and adjusts network weights
   - Target network is updated periodically for stability

6. Performance Tracking
   - Tracks total reward, number of trades, weighted win rate, and average loss per episode
   - Logs detailed metrics after each episode

7. Model Saving
   - Saves the best model based on highest score: "model_best_score_{episode}.pth"
   - Saves the best model based on lowest loss: "model_best_loss_{episode}.pth"
   - Creates periodic checkpoints: "model_checkpoint_{episode}.pth"
   - Saves the final model after training: "model_final.pth"

8. Early Stopping
   - Monitors the average loss over episodes
   - Stops training if no improvement is seen for 50 consecutive episodes

9. Performance Visualization
   - Generates plots for scores, steps per episode, win rates, losses, and epsilon decay
   - Saves plots periodically and at the end of training

10. Model Explanation
    - Attempts to generate SHAP (SHapley Additive exPlanations) values for model interpretability
    - Creates a summary plot of feature importance if successful

11. Evaluation
    - Evaluates the trained model on both validation and test datasets
    - Calculates and logs the mean and standard deviation of scores for each dataset

## Key Training Parameters
- Episodes: 200 (maximum)
- Batch Size: 64
- Learning Rate: 0.0003 (defined in config.py)
- Discount Factor (Gamma): 0.99 (defined in config.py)
- Epsilon: Starts at 1.0, decays by 0.995 after each step, minimum 0.05
- Memory Size: 50000 experiences (defined in config.py)
- Early Stopping Patience: 50 episodes

## Output and Results
All training outputs are saved in the "training_results" folder on the desktop, including:
- Trained models (best score, best loss, checkpoints, and final)
- Performance plots (scores, steps, win rates, losses, epsilon decay)
- SHAP summary plot (if successful)
- Detailed logs of the training process

## Usage
1. Run `database_parser.py` to collect and save historical data
2. Execute `training.py` to train the agent on the collected data
3. Use `main.py` to start the Telegram bot and interact with the trained model

Note: Ensure all required libraries are installed and API keys are properly set in `config.py` before running the scripts.

## Future Improvements
- Implement more sophisticated reward functions
- Explore different network architectures for the DQN
- Incorporate additional features or technical indicators in the state representation
- Implement ensemble methods or multi-agent systems for more robust trading strategies
