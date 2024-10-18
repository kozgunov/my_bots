
# Crypto Trading Bot with Deep Q-Learning

This project implements a cryptocurrency trading bot using Deep Q-Learning, a reinforcement learning technique that combines Q-learning with deep neural networks. The bot is designed to make trading decisions on hourly cryptocurrency data, considering technical indicators and sentiment analysis.

## Features

- Deep Q-Network (DQN) implementation for trading decisions
- Support for multiple data sources (crypto and stocks)
- Technical indicator calculation using TA-Lib
- Sentiment analysis of news headlines
- Customizable trading environment with fees and time constraints
- Data preprocessing and normalization
- Visualization of training progress and trading results

## Project Structure

- `main.py`: The main script to run the trading bot
- `agent.py`: Implementation of the DQN agent
- `environment.py`: Trading environment simulation
- `utils.py`: Utility functions for data loading and preprocessing
- `training.py`: Functions for training and testing the agent
- `Q-learning.py`: Basic Q-learning implementation (for comparison)

## Requirements

- Python 3.7+
- TensorFlow 2.x
- NumPy
- Pandas
- Matplotlib
- scikit-learn
- TA-Lib
- yfinance
- ccxt
- TextBlob

## Installation

1. Clone this repository:   ```
   git clone https://github.com/yourusername/crypto-trading-bot.git
   cd crypto-trading-bot   ```

2. Install the required packages:   ```
   pip install -r requirements.txt   ```

3. Install TA-Lib (follow instructions for your operating system)

## Usage

1. Configure your desired cryptocurrency and parameters in `main.py`
2. Run the main script:   ```
   python main.py   ```

3. The script will train the agent, test it on the most recent data, and generate performance plots.

## Customization

- Adjust the `ImprovedQLearningAgent` parameters in `agent.py` to modify the learning process
- Customize the trading environment in `environment.py` to add more complex trading rules
- Add or remove technical indicators in the `add_technical_indicators` function in `utils.py`

## Results

The bot generates two main output files:
- `training_results.png`: Shows the training progress, including rewards and epsilon decay
- `trading_results.png`: Displays the bot's trading actions and portfolio value over time

## Disclaimer

This project is for educational purposes only. Cryptocurrency trading carries a high level of risk, and this bot should not be used for actual trading without thorough testing and risk assessment.

## Contributing

Contributions, issues, and feature requests are welcome. Feel free to check [issues page](https://github.com/yourusername/crypto-trading-bot/issues) if you want to contribute.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
