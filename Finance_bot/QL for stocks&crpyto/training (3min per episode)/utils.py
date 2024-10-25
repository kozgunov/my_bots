import numpy as np
import ccxt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import requests
from config import TOKEN_NEWS_API, SYMBOL_MAPPING
import warnings
import shap
import matplotlib.pyplot as plt
import os
import torch
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress specific warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, message="divide by zero encountered in divide")
warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered in divide")

def generate_price_data(size=1000, low=100, high=200):
    try:
        return np.random.randint(low, high, size=size)
    except ValueError as e:
        logger.error(f"Error generating price data: {e}")
        return None

def load_crypto_data(exchange_name, symbol, timeframe='1h', limit=1000):
    try:
        exchange = getattr(ccxt, exchange_name)()
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        return df
    except (ccxt.NetworkError, ccxt.ExchangeError, AttributeError) as e:
        logger.error(f"Error loading data from {exchange_name}: {e}")
        return None

def add_technical_indicators(df):
    try:
        df['SMA_20'] = df['close'].rolling(window=20).mean()
        df['SMA_50'] = df['close'].rolling(window=50).mean()
        df['RSI'] = calculate_rsi(df['close'])
        df['MACD'], df['MACD_Signal'] = calculate_macd(df['close'])
        df['ATR'] = calculate_atr(df['high'], df['low'], df['close'])
        df['Bollinger_Middle'], df['Bollinger_Upper'], df['Bollinger_Lower'] = calculate_bollinger_bands(df['close'])
        df['Returns'] = df['close'].pct_change()
        return df
    except Exception as e:
        logger.error(f"Error adding technical indicators: {e}")
        return None

def normalize_data(df):
    try:
        scaler = MinMaxScaler()
        columns_to_normalize = ['open', 'high', 'low', 'close', 'volume', 'SMA_20', 'SMA_50', 'RSI', 'MACD', 'MACD_Signal', 'ATR', 'Bollinger_Upper', 'Bollinger_Middle', 'Bollinger_Lower', 'Returns']
        df[columns_to_normalize] = scaler.fit_transform(df[columns_to_normalize])
        return df, scaler
    except Exception as e:
        logger.error(f"Error normalizing data: {e}")
        return None, None

def prepare_data(df):
    try:
        if df is None or df.empty:
            raise ValueError("Failed to load data or empty DataFrame")

        required_columns = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required_columns):
            raise ValueError("Missing required columns in the DataFrame")

        for col in required_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        df = add_technical_indicators(df)
        df['cci'] = calculate_cci(df['high'], df['low'], df['close'])
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        df['volatility'] = df['log_returns'].rolling(window=30).std() * np.sqrt(252)
        df['volume_ma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma']
        
        df.dropna(inplace=True)

        numeric_columns = df.select_dtypes(include=[np.number]).columns
        df[numeric_columns] = df[numeric_columns].replace([np.inf, -np.inf], np.nan)
        df.dropna(inplace=True)

        if df.empty:
            raise ValueError("No valid data left after preprocessing")

        scaler = MinMaxScaler()
        df[numeric_columns] = scaler.fit_transform(df[numeric_columns])
        
        return df, scaler
    except Exception as e:
        logger.error(f"Error in prepare_data: {e}")
        return None, None

def create_sequences(data, seq_length):
    try:
        xs, ys = [], []
        for i in range(len(data) - seq_length):
            x = data.iloc[i:(i + seq_length)]
            y = data.iloc[i + seq_length]
            xs.append(x)
            ys.append(y)
        return np.array(xs), np.array(ys)
    except Exception as e:
        logger.error(f"Error creating sequences: {e}")
        return None, None

def get_crypto_info(asset):
    try:
        asset_key = asset.lower()
        if asset_key in SYMBOL_MAPPING:
            symbol = SYMBOL_MAPPING[asset_key]
            return f"Crypto info for {symbol}\n"
        else:
            return "Sorry, I couldn't find information about that asset.\n"
    except Exception as e:
        logger.error(f"Error getting crypto info: {e}")
        return "An error occurred while fetching crypto information.\n"

def get_latest_news(asset):
    url = 'https://newsapi.org/v2/everything'
    params = {
        'q': f"{asset} OR cryptocurrency",
        'apiKey': TOKEN_NEWS_API,
        'pageSize': 3,
        'language': 'en',
        'sortBy': 'publishedAt'
    }
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        if data['status'] == 'ok' and data['totalResults'] > 0:
            articles = data['articles']
            news_text = "\nLatest News:\n"
            for i, article in enumerate(articles, 1):
                news_text += f"{i}. [{article['title']}]({article['url']})\n"
            return news_text
        else:
            return f"\nNo recent news found for {asset}.\n"
    except requests.RequestException as e:
        logger.error(f"Error fetching news: {e}")
        return "\nAn error occurred while fetching news.\n"

def calculate_rsi(prices, window=14):
    try:
        delta = np.diff(prices)
        gain = np.maximum(delta, 0)
        loss = -np.minimum(delta, 0)
        
        avg_gain = pd.Series(gain).ewm(com=window-1, adjust=False).mean().values
        avg_loss = pd.Series(loss).ewm(com=window-1, adjust=False).mean().values
        
        avg_gain = np.pad(avg_gain, (1, 0), mode='constant', constant_values=np.nan)
        avg_loss = np.pad(avg_loss, (1, 0), mode='constant', constant_values=np.nan)
        
        rs = np.divide(avg_gain, avg_loss, out=np.ones_like(avg_gain), where=avg_loss!=0)
        rsi = 100 - (100 / (1 + rs))
        
        rsi = np.clip(rsi, 0, 100)
        rsi = np.nan_to_num(rsi, nan=50.0)
        
        return rsi
    except Exception as e:
        logger.error(f"Error calculating RSI: {e}")
        return None

def calculate_macd(prices, fast=12, slow=26, signal=9):
    try:
        prices = pd.Series(prices)
        fast_ema = prices.ewm(span=fast, adjust=False).mean()
        slow_ema = prices.ewm(span=slow, adjust=False).mean()
        macd = fast_ema - slow_ema
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        return macd.values, signal_line.values
    except Exception as e:
        logger.error(f"Error calculating MACD: {e}")
        return None, None

def calculate_bollinger_bands(prices, window=20, num_std=2):
    try:
        prices = pd.Series(prices)
        middle = prices.rolling(window=window).mean()
        std = prices.rolling(window=window).std()
        upper = middle + (std * num_std)
        lower = middle - (std * num_std)
        return middle.values, upper.values, lower.values
    except Exception as e:
        logger.error(f"Error calculating Bollinger Bands: {e}")
        return None, None, None

def calculate_sharpe_ratio(returns, risk_free_rate=0.02):
    try:
        returns = np.array(returns)
        excess_returns = returns - risk_free_rate
        return np.mean(excess_returns) / (np.std(excess_returns) + 1e-10)
    except Exception as e:
        logger.error(f"Error calculating Sharpe Ratio: {e}")
        return None

def calculate_max_drawdown(portfolio_values):
    try:
        portfolio_values = np.array(portfolio_values)
        peak = np.maximum.accumulate(portfolio_values)
        drawdown = (peak - portfolio_values) / peak
        return np.max(drawdown)
    except Exception as e:
        logger.error(f"Error calculating Max Drawdown: {e}")
        return None

def calculate_profit_factor(profits):
    try:
        profits = np.array(profits)
        gains = profits[profits > 0].sum()
        losses = abs(profits[profits < 0].sum())
        return gains / (losses + 1e-10)
    except Exception as e:
        logger.error(f"Error calculating Profit Factor: {e}")
        return None

def plot_shap_values(model, sample_data, feature_names, results_path, episode):
    try:
        sample_tensor = torch.FloatTensor(sample_data).to(model.device)
        explainer = shap.DeepExplainer(model, sample_tensor)
        shap_values = explainer.shap_values(sample_tensor)

        # Summary plot
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, sample_data, feature_names=feature_names, show=False)
        plt.tight_layout()
        plt.savefig(os.path.join(results_path, f"shap_summary_{episode}.png"))
        plt.close()

        # Bar plot
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, sample_data, feature_names=feature_names, plot_type="bar", show=False)
        plt.tight_layout()
        plt.savefig(os.path.join(results_path, f"shap_bar_{episode}.png"))
        plt.close()

        # Force plot for a single prediction
        force_plot = shap.force_plot(explainer.expected_value[0], shap_values[0][0], sample_data[0], feature_names=feature_names, matplotlib=True, show=False)
        plt.tight_layout()
        plt.savefig(os.path.join(results_path, f"shap_force_{episode}.png"))
        plt.close()

        return shap_values, explainer.expected_value
    except Exception as e:
        logger.error(f"Error plotting SHAP values: {e}")
        return None, None

def calculate_atr(high, low, close, period=14):
    try:
        tr = np.maximum(high - low, np.abs(high - close.shift(1)), np.abs(low - close.shift(1)))
        return tr.rolling(window=period).mean()
    except Exception as e:
        logger.error(f"Error calculating ATR: {e}")
        return None

def calculate_cci(high, low, close, period=20):
    try:
        tp = (high + low + close) / 3
        sma = tp.rolling(window=period).mean()
        mad = tp.rolling(window=period).apply(lambda x: np.abs(x - x.mean()).mean())
        return (tp - sma) / (0.015 * mad)
    except Exception as e:
        logger.error(f"Error calculating CCI: {e}")
        return None
