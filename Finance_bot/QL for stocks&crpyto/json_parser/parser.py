import json
import logging
from datetime import datetime, timedelta
import requests
import os
from config import BINANCE_API_KEY, BINANCE_SECRET_KEY

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Binance API endpoint
BASE_URL = "https://api.binance.com/api/v3"

# List of 10 popular coins
COINS = ['BTC', 'ETH', 'BNB', 'ADA', 'XRP', 'DOT', 'UNI', 'LTC', 'LINK', 'SOL']

# Timeframe
INTERVAL = "15m"

# Number of data points to fetch (approximately 4 years of 15-minute data)
LIMIT = 365 * 24 * 4 * 4  # 4 years * 365 days * 24 hours * 4 (15-minute intervals per hour)

# Get the path to the desktop
DESKTOP_PATH = os.path.join(os.path.expanduser("~"), "Desktop")

def fetch_and_process_data(symbol):
    logger.info(f"Fetching data for {symbol}...")
    
    try:
        all_data = []
        end_time = int(datetime.now().timestamp() * 1000)
        
        while len(all_data) < LIMIT:
            # Construct the API request URL
            endpoint = f"{BASE_URL}/klines"
            params = {
                "symbol": f"{symbol}USDT",
                "interval": INTERVAL,
                "limit": 1000,  # Binance API limit
                "endTime": end_time
            }
            headers = {
                "X-MBX-APIKEY": BINANCE_API_KEY
            }
            
            # Make the API request
            response = requests.get(endpoint, params=params, headers=headers)
            response.raise_for_status()  # Raise an exception for bad responses
            klines = response.json()
            
            if not klines:
                break
            
            all_data = klines + all_data
            end_time = klines[0][0] - 1  # Set end time to the oldest timestamp minus 1 millisecond
        
        processed_data = []
        for kline in all_data[:LIMIT]:
            timestamp, open_price, high, low, close, volume = kline[:6]
            
            # Convert timestamp to readable format
            date = datetime.fromtimestamp(int(timestamp) / 1000).strftime('%Y-%m-%d %H:%M:%S')
            
            processed_data.append({
                'timestamp': date,
                'coin': symbol,
                'open': float(open_price),
                'high': float(high),
                'low': float(low),
                'close': float(close),
                'volume': float(volume)
            })
        
        logger.info(f"Successfully processed {len(processed_data)} records for {symbol}")
        return processed_data
    except requests.exceptions.RequestException as e:
        logger.error(f"API request error for {symbol}: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error for {symbol}: {e}")
        return None

def save_to_json(data, filename):
    try:
        filepath = os.path.join(DESKTOP_PATH, filename)
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        logger.info(f"Successfully saved data to {filepath}")
    except Exception as e:
        logger.error(f"Error saving data to {filepath}: {e}")

def test_data_retrieval():
    test_coin = 'BTC'
    test_data = fetch_and_process_data(test_coin)
    if test_data and len(test_data) > 0:
        logger.info(f"Test successful. Retrieved {len(test_data)} records for {test_coin}")
        logger.info(f"Sample data: {test_data[0]}")
        logger.info(f"Date range: from {test_data[-1]['timestamp']} to {test_data[0]['timestamp']}")
    else:
        logger.error("Test failed. Unable to retrieve data.")

def main():
    logger.info("Starting data collection process")
    all_data = []
    
    for coin in COINS:
        coin_data = fetch_and_process_data(coin)
        if coin_data:
            all_data.extend(coin_data)
            
            # Save individual coin data to desktop
            save_to_json(coin_data, f"{coin}_data.json")
            logger.info(f"Saved {len(coin_data)} records for {coin}")
        else:
            logger.warning(f"Skipping {coin} due to data retrieval failure")
    
    # Save all data combined to desktop
    save_to_json(all_data, "all_coins_data.json")
    logger.info(f"Saved {len(all_data)} total records for all coins")

if __name__ == "__main__":
    test_data_retrieval()
    main()
