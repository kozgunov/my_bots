import numpy as np
import ccxt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta
import requests
from config import TOKEN_NEWS_API, SYMBOL_MAPPING

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
    """
    Prepare cryptocurrency data for trading.
    
    Args:
    df (pandas.DataFrame): DataFrame with OHLCV data
    
    Returns:
    tuple: (prepared_df, scaler)
    """
    if df is None:
        print("Failed to load data")
        return None, None

    df = add_technical_indicators(df)
    df, scaler = normalize_data(df)
    
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
