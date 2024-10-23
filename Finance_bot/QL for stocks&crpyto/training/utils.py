import numpy as np
import ccxt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta
import requests
from config import TOKEN_NEWS_API, SYMBOL_MAPPING
import warnings

# Suppress specific warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, message="divide by zero encountered in divide")
warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered in divide")

def generate_price_data(size=1000, low=100, high=200):
    """
    Generate dummy price data for testing purposes.
    
    Args:
    size (int): Number of data points to generate
    low (int): Minimum price value
    high (int): Maximum price value
    
    Returns:
    numpy.array: Array of randomly generated price data
    """
    return np.random.randint(low, high, size=size)

def load_crypto_data(exchange_name, symbol, timeframe='1h', limit=1000):
    """
    Load cryptocurrency data from a specified exchange.
    
    Args:
    exchange_name (str): Name of the exchange (e.g., 'binance', 'okx')
    symbol (str): Trading pair symbol (e.g., 'BTC/USDT')
    timeframe (str): Data timeframe (e.g., '1d' for daily, '1h' for hourly)
    limit (int): Number of data points to retrieve
    
    Returns:
    pandas.DataFrame: DataFrame with OHLCV data, or None if an error occurs
    """
    try:
        exchange = getattr(ccxt, exchange_name)()
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        return df
    except Exception as e:
        print(f"Error loading data from {exchange_name}: {e}")
        return None

def add_technical_indicators(df):
    """
    Add technical indicators to the DataFrame.
    
    Args:
    df (pandas.DataFrame): DataFrame with OHLCV data
    
    Returns:
    pandas.DataFrame: DataFrame with added technical indicators
    """
    # Simple Moving Average (SMA)
    df['SMA_20'] = df['close'].rolling(window=20).mean()
    df['SMA_50'] = df['close'].rolling(window=50).mean()

    # Relative Strength Index (RSI)
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # Moving Average Convergence Divergence (MACD)
    exp1 = df['close'].ewm(span=12, adjust=False).mean()
    exp2 = df['close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

    # Average True Range (ATR)
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    df['ATR'] = true_range.rolling(14).mean()

    # Bollinger Bands
    df['Bollinger_Middle'] = df['close'].rolling(window=20).mean()
    df['Bollinger_Std'] = df['close'].rolling(window=20).std()
    df['Bollinger_Upper'] = df['Bollinger_Middle'] + (df['Bollinger_Std'] * 2)
    df['Bollinger_Lower'] = df['Bollinger_Middle'] - (df['Bollinger_Std'] * 2)

    # Returns
    df['Returns'] = df['close'].pct_change()

    return df

def normalize_data(df):
    """
    Normalize the data using MinMaxScaler.
    
    Args:
    df (pandas.DataFrame): DataFrame with OHLCV and technical indicator data
    
    Returns:
    tuple: (normalized_df, scaler)
    """
    scaler = MinMaxScaler()
    columns_to_normalize = ['open', 'high', 'low', 'close', 'volume', 'SMA_20', 'SMA_50', 'RSI', 'MACD', 'MACD_Signal', 'ATR', 'Bollinger_Upper', 'Bollinger_Middle', 'Bollinger_Lower', 'Returns']
    df[columns_to_normalize] = scaler.fit_transform(df[columns_to_normalize])
    return df, scaler

def prepare_data(df):
    if df is None or df.empty:
        print("Failed to load data or empty DataFrame")
        return None, None

    # Ensure all necessary columns are present
    required_columns = ['open', 'high', 'low', 'close', 'volume']
    if not all(col in df.columns for col in required_columns):
        print("Missing required columns in the DataFrame")
        return None, None

    # Convert columns to numeric, coercing errors to NaN
    for col in required_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Add technical indicators
    df['sma_10'] = df['close'].rolling(window=10).mean()
    df['sma_30'] = df['close'].rolling(window=30).mean()
    df['rsi'] = calculate_rsi(df['close'].values)
    macd, signal = calculate_macd(df['close'].values)
    df['macd'] = macd
    df['signal'] = signal
    middle, upper, lower = calculate_bollinger_bands(df['close'].values)
    df['bollinger_middle'] = middle
    df['bollinger_upper'] = upper
    df['bollinger_lower'] = lower

    # Remove any rows with NaN values
    df.dropna(inplace=True)

    # Check for infinite values
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    df[numeric_columns] = df[numeric_columns].replace([np.inf, -np.inf], np.nan)
    df.dropna(inplace=True)

    if df.empty:
        print("No valid data left after preprocessing")
        return None, None

    # Normalize the data
    scaler = MinMaxScaler()
    df[numeric_columns] = scaler.fit_transform(df[numeric_columns])
    
    return df, scaler

def create_sequences(data, seq_length):
    """
    Create input sequences and corresponding targets for time series prediction.
    
    Args:
    data (pandas.DataFrame): Input data
    seq_length (int): Length of input sequences
    
    Returns:
    tuple: (input_sequences, targets) as numpy arrays
    """
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data.iloc[i:(i + seq_length)]
        y = data.iloc[i + seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

def get_crypto_info(asset):
    asset_key = asset.lower()
    if asset_key in SYMBOL_MAPPING:
        symbol = SYMBOL_MAPPING[asset_key]
    else:
        return "Sorry, I couldn't find information about that asset.\n"

    # Implement the logic to fetch crypto info
    # This is a placeholder, implement the actual API calls
    return f"Crypto info for {symbol}\n"

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
    except Exception as e:
        print(f"Error fetching news: {e}")
        return "\nAn error occurred while fetching news.\n"

def calculate_rsi(prices, window=14):
    delta = np.diff(prices)
    gain = np.maximum(delta, 0)
    loss = -np.minimum(delta, 0)
    
    # Use exponential moving average for more stability
    avg_gain = pd.Series(gain).ewm(com=window-1, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(com=window-1, adjust=False).mean().values
    
    # Pad the beginning to match the original length
    avg_gain = np.pad(avg_gain, (1, 0), mode='constant', constant_values=np.nan)
    avg_loss = np.pad(avg_loss, (1, 0), mode='constant', constant_values=np.nan)
    
    # Handle division by zero and invalid values
    rs = np.divide(avg_gain, avg_loss, out=np.ones_like(avg_gain), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    # Replace infinity, NaN, and out-of-bounds values
    rsi = np.clip(rsi, 0, 100)
    rsi = np.nan_to_num(rsi, nan=50.0)
    
    return rsi

def calculate_macd(prices, fast=12, slow=26, signal=9):
    if isinstance(prices, np.ndarray):
        prices = pd.Series(prices)
    
    fast_ema = prices.ewm(span=fast, adjust=False).mean()
    slow_ema = prices.ewm(span=slow, adjust=False).mean()
    macd = fast_ema - slow_ema
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd.values, signal_line.values

def calculate_bollinger_bands(prices, window=20, num_std=2):
    if isinstance(prices, np.ndarray):
        prices = pd.Series(prices)
    
    middle = prices.rolling(window=window).mean()
    std = prices.rolling(window=window).std()
    upper = middle + (std * num_std)
    lower = middle - (std * num_std)
    return middle.values, upper.values, lower.values
