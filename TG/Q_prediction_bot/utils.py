import numpy as np
import ccxt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import talib
from datetime import datetime, timedelta

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
    df['SMA_20'] = talib.SMA(df['close'], timeperiod=20)
    df['SMA_50'] = talib.SMA(df['close'], timeperiod=50)
    df['RSI'] = talib.RSI(df['close'], timeperiod=14)
    df['MACD'], df['MACD_Signal'], _ = talib.MACD(df['close'], fastperiod=12, slowperiod=26, signalperiod=9)
    df['ATR'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
    df['Bollinger_Upper'], df['Bollinger_Middle'], df['Bollinger_Lower'] = talib.BBANDS(df['close'], timeperiod=20)
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

def prepare_data(exchange_name, symbol, timeframe='1h', limit=1000):
    """
    Load and prepare cryptocurrency data for trading.
    
    Args:
    exchange_name (str): Name of the exchange
    symbol (str): Trading pair symbol
    timeframe (str): Data timeframe
    limit (int): Number of data points to retrieve
    
    Returns:
    tuple: (prepared_df, scaler) or (None, None) if data loading fails
    """
    df = load_crypto_data(exchange_name, symbol, timeframe, limit)
    
    if df is None:
        print(f"Failed to load data for {symbol}")
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
