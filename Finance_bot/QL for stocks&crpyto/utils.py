import numpy as np
import ccxt
import pandas as pd
import yfinance as yf
from alpaca_trade_api.rest import REST
import tensortrade.env.default as default
from sklearn.preprocessing import MinMaxScaler
import talib
from textblob import TextBlob
from datetime import datetime, timedelta

def generate_price_data(size=1000, low=100, high=200):
    return np.random.randint(low, high, size=size)

def preprocess_data(data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data.reshape(-1, 1))
    return scaled_data.flatten(), scaler

def load_crypto_data(exchange_name, symbol, timeframe='1d', limit=1000):
    try:
        # Initialize the exchange
        exchange = getattr(ccxt, exchange_name)()
        
        # Fetch OHLCV data
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        
        # Convert to DataFrame
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        
        # Return only the 'close' prices as a numpy array
        return df['close'].values
    except Exception as e:
        print(f"Error loading data from {exchange_name}: {e}")
        return None

def load_okx_data(symbol='BTC/USDT', timeframe='1d', limit=1000):
    return load_crypto_data('okx', symbol, timeframe, limit)

def load_bybit_data(symbol='BTC/USDT', timeframe='1d', limit=1000):
    return load_crypto_data('bybit', symbol, timeframe, limit)

def load_yfinance_data(symbol, period="1000d", interval="1d"):
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period, interval=interval)
        return df['Close'].values
    except Exception as e:
        print(f"Error loading data from yfinance: {e}")
        return None

def load_alpaca_data(symbol, timeframe='1D', limit=1000):
    try:
        api = REST('YOUR_API_KEY', 'YOUR_SECRET_KEY', base_url='https://paper-api.alpaca.markets')
        bars = api.get_barset(symbol, timeframe, limit=limit)[symbol]
        return [bar.c for bar in bars]  # Return closing prices
    except Exception as e:
        print(f"Error loading data from Alpaca: {e}")
        return None

def load_tensortrade_data(symbol, exchange="bitfinex", start="2010-01-01", end="2023-05-01"):
    try:
        data_loader = default.data.CSVDataLoader(
            filename=f"path/to/{exchange}_{symbol}.csv",
            base_instrument="USD",
            quote_instrument=symbol,
            start_date=start,
            end_date=end
        )
        data = data_loader.load()
        return data['close'].values
    except Exception as e:
        print(f"Error loading data from TensorTrade: {e}")
        return None

def load_real_data(source, symbol, **kwargs):
    data_loaders = {
        'okx': load_okx_data,
        'bybit': load_bybit_data,
        'yfinance': load_yfinance_data,
        'alpaca': load_alpaca_data,
        'tensortrade': load_tensortrade_data
    }
    
    loader = data_loaders.get(source.lower())
    if loader:
        data = loader(symbol, **kwargs)
        if data is not None:
            return preprocess_data(data)
        else:
            print(f"Failed to load data from {source}. Using dummy data.")
    else:
        print(f"Unsupported data source: {source}. Using dummy data.")
    
    dummy_data = generate_price_data(1000)
    return preprocess_data(dummy_data)

def load_and_prepare_data(symbol, is_crypto=False, period="60d", interval="1h"):
    if is_crypto:
        df = load_crypto_data(symbol, period, interval)
    else:
        df = load_stock_data(symbol, period, interval)
    
    if df is None:
        print(f"Failed to load data for {symbol}")
        return None, None

    df = add_technical_indicators(df)
    df = add_sentiment_scores(symbol, df)
    df = handle_gaps(df, is_crypto)
    df, scaler = normalize_data(df)
    
    return df, scaler

def load_crypto_data(symbol, period, interval):
    try:
        exchange = ccxt.binance()
        timeframe = interval
        since = exchange.parse8601((datetime.now() - timedelta(days=int(period[:-1]))).isoformat())
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        return df
    except Exception as e:
        print(f"Error loading crypto data: {e}")
        return None

def load_stock_data(symbol, period, interval):
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period, interval=interval)
        return df
    except Exception as e:
        print(f"Error loading stock data: {e}")
        return None

def add_technical_indicators(df):
    df['SMA_20'] = talib.SMA(df['close'], timeperiod=20)
    df['SMA_50'] = talib.SMA(df['close'], timeperiod=50)
    df['RSI'] = talib.RSI(df['close'], timeperiod=14)
    df['MACD'], df['MACD_Signal'], _ = talib.MACD(df['close'], fastperiod=12, slowperiod=26, signalperiod=9)
    df['ATR'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
    df['Bollinger_Upper'], df['Bollinger_Middle'], df['Bollinger_Lower'] = talib.BBANDS(df['close'], timeperiod=20)
    df['Returns'] = df['close'].pct_change()
    return df

def add_sentiment_scores(symbol, df):
    news = yf.Ticker(symbol).news
    sentiment_scores = {}
    
    for article in news:
        date = pd.to_datetime(article['providerPublishTime'], unit='s')
        sentiment = TextBlob(article['title']).sentiment.polarity
        if date in sentiment_scores:
            sentiment_scores[date].append(sentiment)
        else:
            sentiment_scores[date] = [sentiment]
    
    for date in sentiment_scores:
        sentiment_scores[date] = np.mean(sentiment_scores[date])
    
    df['Sentiment'] = df.index.map(lambda x: sentiment_scores.get(x, 0))
    df['Sentiment'] = df['Sentiment'].fillna(method='ffill')
    return df

def handle_gaps(df, is_crypto):
    if not is_crypto:
        # For stocks, forward fill the last known price during market closures
        df = df.resample('1H').ffill()
    
    # Add a column to indicate the time since last update
    df['TimeSinceLastUpdate'] = (df.index - df.index.shift(1)).seconds / 3600
    
    return df

def normalize_data(df):
    scaler = MinMaxScaler()
    columns_to_normalize = ['open', 'high', 'low', 'close', 'volume', 'SMA_20', 'SMA_50', 'RSI', 'MACD', 'MACD_Signal', 'ATR', 'Bollinger_Upper', 'Bollinger_Middle', 'Bollinger_Lower', 'Returns', 'Sentiment']
    df[columns_to_normalize] = scaler.fit_transform(df[columns_to_normalize])
    return df, scaler

def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data.iloc[i:(i + seq_length)]
        y = data.iloc[i + seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

def get_sentiment_scores(ticker, dates):
    news = yf.Ticker(ticker).news
    daily_sentiment = {}
    
    for article in news:
        date = pd.to_datetime(article['providerPublishTime'], unit='s').date()
        sentiment = TextBlob(article['title']).sentiment.polarity
        if date in daily_sentiment:
            daily_sentiment[date].append(sentiment)
        else:
            daily_sentiment[date] = [sentiment]
    
    for date in daily_sentiment:
        daily_sentiment[date] = np.mean(daily_sentiment[date])
    
    return pd.Series([daily_sentiment.get(date.date(), 0) for date in dates], index=dates)
