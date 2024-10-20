# Crypto Trading Bot with Deep Q-Learning (PyTorch)

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
