import telebot
from telebot.types import ReplyKeyboardMarkup, KeyboardButton
import requests
import logging
from config import TOKEN_TG, TOKEN_NEWS_API, TOKEN_FMP, TOKEN_INTRINIO, TOKEN_BYBIT, TOKEN_OKX
import time
from functools import wraps
import psycopg2
from datetime import datetime
import json


BOT_TOKEN = TOKEN_TG
NEWS_API_KEY = TOKEN_NEWS_API
bot = telebot.TeleBot(BOT_TOKEN)

SYMBOL_MAPPING = {
    'bitcoin': 'BTC-USDT',
    'btc': 'BTC-USDT',
    'ethereum': 'ETH-USDT',
    'eth': 'ETH-USDT',
    'ripple': 'XRP-USDT',
    'xrp': 'XRP-USDT',
}

SUGGESTIONS = {
    'bitcoin': ['ethereum', 'litecoin', 'ripple'],
    'btc': ['ethereum', 'litecoin', 'ripple'],
    'ethereum': ['bitcoin', 'cardano', 'polkadot'],
    'eth': ['bitcoin', 'cardano', 'polkadot'],
}

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)


# menu & buttons
def create_main_keyboard():
    keyboard = ReplyKeyboardMarkup(resize_keyboard=True)
    keyboard.row(
        KeyboardButton("Show Current Stocks"),
        KeyboardButton("Show Current Crypto")
    )
    keyboard.row(
        KeyboardButton("Get My Data"),
        KeyboardButton("Remove My Data")
    )
    return keyboard


# show all opportunities of bot
@bot.message_handler(commands=['start', 'help'])
def send_welcome(message):
    help_text = (
        "Welcome to the Crypto Info Bot! Here are the commands you can use:\n"
        "/help - Show this help message\n"
        "/get_my_data - Retrieve your stored data\n"
        "/remove_my_data - Remove your stored data\n"
        "/show_current_crypto - Show current cryptocurrency prices\n"
        "/show_current_stock - Show current stock prices\n"
    )
    bot.send_message(message.chat.id, help_text, reply_markup=create_main_keyboard())


# show some data from user
@bot.message_handler(commands=['get_my_data'])
def get_my_data_command(message):
    get_user_data(message)


# remove all data of user, corresponding to user_id
@bot.message_handler(commands=['remove_my_data'])
def remove_my_data_command(message):
    remove_user_data(message)


# request and show the latest of the latest crypto news
@bot.message_handler(commands=['show_current_crypto'])
def show_current_crypto_command(message):
    show_current_crypto(message)


# request and show the latest of the latest stock news
@bot.message_handler(commands=['show_current_stock'])
def show_current_stock_command(message):
    show_current_stocks(message)


# 3 headers of chosen stocks and attached links
def show_current_stocks(message):
    stocks = ["AAPL", "GOOGL", "MSFT"]  # 3 stocks, ----------fix with user's input--------------
    response = "Current Stock Information:\n"
    for stock in stocks:
        stock_info = get_stock_info(stock)
        response += stock_info + "\n"
    response += get_latest_news("stocks")
    bot.send_message(message.chat.id, response, parse_mode='Markdown', reply_markup=create_main_keyboard())


# 3 headers of chosen coins and attached links
def show_current_crypto(message):
    cryptos = ["BTC", "ETH", "XRP"] # 3 coins, ----------fix with user's input--------------
    response = "Current Cryptocurrency Information:\n\n"
    for crypto in cryptos:
        crypto_info = get_crypto_info(crypto)
        response += crypto_info + "\n"
    response += get_latest_news("cryptocurrency")
    bot.send_message(message.chat.id, response, parse_mode='Markdown', reply_markup=create_main_keyboard())

# show to user only small part of data
def get_user_data(message):
    user_id = message.from_user.id
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM current_price_bot WHERE user_id = %s", (user_id,)) # constructed request
    user_data = cur.fetchone()
    conn.close()

    if user_data:
        response = (
            f"User ID: {user_data[0]}\n"
            f"Username: {user_data[1]}\n"
            f"First Name: {user_data[2]}\n"
            f"Last Name: {user_data[3]}\n"
            f"Language Code: {user_data[4]}\n"
            f"Last Input: {user_data[5]}\n"
            f"Last Activity: {user_data[6]}\n"
        )
    else:
        response = "No data found for your user ID."
    
    bot.send_message(message.chat.id, response, reply_markup=create_main_keyboard())


# find and remove user's data
def remove_user_data(message):
    user_id = message.from_user.id
    conn = get_db_connection()
    cur = conn.cursor()
    
    cur.execute("SELECT * FROM current_price_bot WHERE user_id = %s", (user_id,)) # %s for safety, just filler
    user_data = cur.fetchone()
    
    if user_data:
        cur.execute("DELETE FROM current_price_bot WHERE user_id = %s", (user_id,))
        conn.commit()
        response = "Your data has been removed. Here's what was deleted: " + ", ".join(map(str, user_data))
    else:
        response = "No data found for your user ID."
    
    conn.close()
    bot.send_message(message.chat.id, response, reply_markup=create_main_keyboard())


# Ddos-safety by limitation of requests
def rate_limit(limit_per_minute):
    min_interval = 60.0 / limit_per_minute
    last_called = [0.0]

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            elapsed = time.time() - last_called[0]
            left_to_wait = min_interval - elapsed
            if left_to_wait > 0:
                time.sleep(left_to_wait)
            ret = func(*args, **kwargs)
            last_called[0] = time.time()
            return ret
        return wrapper
    return decorator


# correct gathering of data, exceptions are handled
@rate_limit(60)  # Limit to 60 calls per minute
@bot.message_handler(func=lambda message: True)
def handle_message(message):
    user_input = message.text.lower()
    start_time = time.time()
    
    user_data = {
        'user_id': message.from_user.id,
        'username': message.from_user.username,
        'first_name': message.from_user.first_name,
        'last_name': message.from_user.last_name,
        'language_code': message.from_user.language_code,
        'is_premium': message.from_user.is_premium,
        'is_bot': message.from_user.is_bot,
        'last_input': user_input,
        'last_activity': datetime.now().isoformat(),
        'device_info': message.from_user.language_code, # doesn't work
        'last_location': {'latitude': message.location.latitude, 'longitude': message.location.longitude} if message.location else {}, # doesn't work
    }
    
    if user_input == "show current stocks":
        show_current_stocks(message)
    elif user_input == "show current crypto":
        show_current_crypto(message)
    elif user_input == "get my data":
        get_user_data(message)
    elif user_input == "remove my data":
        remove_user_data(message)
        return  # don't update user data after removal
    else:
        if user_input.isupper() and len(user_input) <= 5:
            response = get_stock_info(user_input)
        else:
            response = get_asset_info(user_input)
        
        bot.send_message(
            message.chat.id,
            response,
            parse_mode='Markdown',
            disable_web_page_preview=True,
            reply_markup=create_main_keyboard()
        )
    
    end_time = time.time()
    user_data['response_time'] = end_time - start_time
    user_data['query_successful'] = True  # set this, if request was correctly applied
    
    update_user_data(user_data)
    print(f"User data updated: {user_data}")


# exception handler
def get_asset_info(asset):
    response = f"I found the latest information about *{asset.title()}*:\n\n"
    crypto_info = get_crypto_info(asset)
    if not crypto_info:
        return f"Sorry, I couldn't find any information about {asset.title()} from the supported exchanges."
    response += crypto_info
    news = get_latest_news(asset)
    response += news if news else "\nNo relevant news found at the moment.\n"
    response += suggest_similar_assets(asset)
    return response


# exception handler
def get_crypto_info(asset):
    asset_key = asset.lower()
    if asset_key in SYMBOL_MAPPING:
        symbol_okx = SYMBOL_MAPPING[asset_key]
        symbol_bybit = symbol_okx.replace('-', '')
    else:
        return "Sorry, I couldn't find information about that asset.\n"

    okx_data = get_okx_data(symbol_okx)
    bybit_data = get_bybit_data(symbol_bybit)

    response_text = ""

    if okx_data:
        response_text += f"1. Current Price on [OKX](https://www.okx.com/): ${okx_data['last']:.2f}\n"
        response_text += f"   24h Change: {okx_data['changePercentage']:.2f}%\n"
    else:
        response_text += "Data from OKX is not available.\n"

    if bybit_data:
        response_text += f"2. Current Price on [Bybit](https://www.bybit.com/): ${bybit_data['lastPrice']}\n"
        response_text += f"   24h Change: {bybit_data['price24hPcnt']:.2f}%\n"
    else:
        response_text += "Data from Bybit is not available.\n"

    return response_text


# gathering data for OKX via API
def get_okx_data(symbol):
    url = 'https://www.okx.com/api/v5/market/ticker'
    params = {'instId': symbol}
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        if data['code'] == '0' and data['data']:
            ticker = data['data'][0]
            last_price = float(ticker['last'])
            open_24h = float(ticker['open24h'])
            change_percentage = ((last_price - open_24h) / open_24h) * 100
            return {
                'last': last_price,
                'changePercentage': change_percentage
            }
        else:
            logger.warning(f"Unexpected response from OKX: {data}")
            return None
    except requests.RequestException as e:
        logger.error(f"Error fetching data from OKX: {e}")
        return None
    except (KeyError, ValueError) as e:
        logger.error(f"Error parsing OKX data: {e}")
        return None


# gathering data for Bybit via API
def get_bybit_data(symbol):
    url = 'https://api.bybit.com/v5/market/tickers'
    params = {'category': 'spot', 'symbol': symbol}
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        if data['retCode'] == 0 and data['result']['list']:
            ticker = data['result']['list'][0]
            return {
                'lastPrice': ticker['lastPrice'],
                'price24hPcnt': float(ticker['price24hPcnt']) * 100
            }
        else:
            logger.warning(f"Unexpected response from Bybit: {data}")
            return None
    except requests.RequestException as e:
        logger.error(f"Error fetching data from Bybit: {e}")
        return None
    except (KeyError, ValueError) as e:
        logger.error(f"Error parsing Bybit data: {e}")
        return None


# gathering latest news from NewsAPI and mentioned stocks/crypto
def get_latest_news(asset):
    url = 'https://newsapi.org/v2/everything'
    params = {
        'q': f"{asset} OR stock OR cryptocurrency",
        'apiKey': NEWS_API_KEY,
        'pageSize': 5,  # shows 3, for 5 it has more chances of finding valid articles
        'language': 'en',
        'sortBy': 'publishedAt'
    }
    try:
        response = requests.get(url, params=params, timeout=5)
        response.raise_for_status()
        data = response.json()
        if data['status'] == 'ok' and data['totalResults'] > 0:
            articles = data['articles']
            response_text = "\nLatest News:\n"
            valid_articles = 0
            for i, article in enumerate(articles, 1):
                title = article.get('title', '').strip()
                url = article.get('url', '').strip()
                if title and url and 'removed' not in url.lower():
                    response_text += f"{valid_articles + 1}. [{title}]({url})\n"
                    valid_articles += 1
                    if valid_articles == 3:
                        break
            if valid_articles > 0:
                return response_text
            else:
                return "\nNo valid news articles found at the moment.\n"
        else:
            return f"\nNo recent news found for {asset}.\n"
    except requests.RequestException as e:
        logger.error(f"Error fetching news: {e}")
        return "\nAn error occurred while fetching news.\n"


# recommendation for queries
def suggest_similar_assets(asset):
    asset_key = asset.lower()
    if asset_key in SUGGESTIONS:
        suggested = SUGGESTIONS[asset_key]
        response_text = "\nYou might also be interested in:\n"
        for s in suggested:
            response_text += f"- {s.title()}\n"
        return response_text
    else:
        return ""


# connection to database
def get_db_connection():
    return psycopg2.connect(
        dbname="postgres",
        user="postgres",
        password=1111,
        host="localhost",
        port=5432
    )


# it will be created in the 1st start only!
def create_current_price_bot_table():
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS current_price_bot (
            user_id BIGINT PRIMARY KEY NOT NULL,
            username TEXT,
            first_name TEXT,
            last_name TEXT,
            language_code TEXT,
            last_input TEXT,
            last_activity TIMESTAMP NOT NULL,
            registration_date TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            total_messages INT DEFAULT 0,
            favorite_crypto TEXT,
            favorite_stock TEXT,
            preferred_currency TEXT DEFAULT 'USD',
            notification_settings JSONB,
            last_location JSONB,
            device_info TEXT,
            referral_code TEXT,
            referred_by BIGINT,
            subscription_tier TEXT,
            subscription_expiry TIMESTAMP,
            total_queries INT DEFAULT 0,
            successful_queries INT DEFAULT 0,
            failed_queries INT DEFAULT 0,
            average_response_time FLOAT,
            last_feedback TEXT,
            custom_settings JSONB
        )
    """)
    conn.commit()
    conn.close()


# download data from user
def update_user_data(user_data):
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO current_price_bot 
        (user_id, username, first_name, last_name, language_code, is_premium, is_bot, 
        last_input, last_activity, total_messages, favorite_crypto, favorite_stock, 
        preferred_currency, notification_settings, last_location, device_info, 
        total_queries, successful_queries, failed_queries, average_response_time, 
        last_feedback, custom_settings)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (user_id) DO UPDATE SET  
            username = EXCLUDED.username,
            first_name = EXCLUDED.first_name,
            last_name = EXCLUDED.last_name,
            language_code = EXCLUDED.language_code,
            is_premium = EXCLUDED.is_premium,
            is_bot = EXCLUDED.is_bot,
            last_input = EXCLUDED.last_input,
            last_activity = EXCLUDED.last_activity,
            total_messages = current_price_bot.total_messages + 1,
            favorite_crypto = COALESCE(EXCLUDED.favorite_crypto, current_price_bot.favorite_crypto),
            favorite_stock = COALESCE(EXCLUDED.favorite_stock, current_price_bot.favorite_stock),
            preferred_currency = COALESCE(EXCLUDED.preferred_currency, current_price_bot.preferred_currency),
            notification_settings = COALESCE(EXCLUDED.notification_settings, current_price_bot.notification_settings),
            last_location = COALESCE(EXCLUDED.last_location, current_price_bot.last_location),
            device_info = COALESCE(EXCLUDED.device_info, current_price_bot.device_info),
            total_queries = current_price_bot.total_queries + 1,
            successful_queries = current_price_bot.successful_queries + CASE WHEN EXCLUDED.successful_queries > 0 THEN 1 ELSE 0 END,
            failed_queries = current_price_bot.failed_queries + CASE WHEN EXCLUDED.failed_queries > 0 THEN 1 ELSE 0 END,
            average_response_time = (current_price_bot.average_response_time * current_price_bot.total_queries + EXCLUDED.average_response_time) / (current_price_bot.total_queries + 1),
            last_feedback = COALESCE(EXCLUDED.last_feedback, current_price_bot.last_feedback),
            custom_settings = COALESCE(EXCLUDED.custom_settings, current_price_bot.custom_settings)
    """, (
        user_data['user_id'],
        user_data['username'],
        user_data['first_name'],
        user_data['last_name'],
        user_data['language_code'],
        user_data.get('is_premium', False),
        user_data.get('is_bot', False),
        user_data['last_input'],
        user_data['last_activity'],
        1,  # increment total_messages
        user_data.get('favorite_crypto'),
        user_data.get('favorite_stock'),
        user_data.get('preferred_currency', 'USD'),
        json.dumps(user_data.get('notification_settings', {})),
        json.dumps(user_data.get('last_location', {})),
        user_data.get('device_info'),
        1,  # increment total_queries
        1 if user_data.get('query_successful', True) else 0,
        0 if user_data.get('query_successful', True) else 1,
        user_data.get('response_time', 0),
        user_data.get('last_feedback'),
        json.dumps(user_data.get('custom_settings', {}))
    ))
    conn.commit()
    conn.close()


# update database with this user
def update_database_schema():
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        cur.execute("""
        ALTER TABLE current_price_bot
        ADD COLUMN IF NOT EXISTS is_premium BOOLEAN,
        ADD COLUMN IF NOT EXISTS is_bot BOOLEAN,
        ADD COLUMN IF NOT EXISTS registration_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        ADD COLUMN IF NOT EXISTS total_messages INT DEFAULT 0,
        ADD COLUMN IF NOT EXISTS favorite_crypto TEXT,
        ADD COLUMN IF NOT EXISTS favorite_stock TEXT,
        ADD COLUMN IF NOT EXISTS preferred_currency TEXT DEFAULT 'USD',
        ADD COLUMN IF NOT EXISTS notification_settings JSONB,
        ADD COLUMN IF NOT EXISTS last_location JSONB,
        ADD COLUMN IF NOT EXISTS device_info TEXT,
        ADD COLUMN IF NOT EXISTS referral_code TEXT,
        ADD COLUMN IF NOT EXISTS referred_by BIGINT,
        ADD COLUMN IF NOT EXISTS subscription_tier TEXT,
        ADD COLUMN IF NOT EXISTS subscription_expiry TIMESTAMP,
        ADD COLUMN IF NOT EXISTS total_queries INT DEFAULT 0,
        ADD COLUMN IF NOT EXISTS successful_queries INT DEFAULT 0,
        ADD COLUMN IF NOT EXISTS failed_queries INT DEFAULT 0,
        ADD COLUMN IF NOT EXISTS average_response_time FLOAT,
        ADD COLUMN IF NOT EXISTS last_feedback TEXT,
        ADD COLUMN IF NOT EXISTS custom_settings JSONB
        """)
        conn.commit()
        print("Database schema updated successfully.")
    except Exception as e:
        print(f"Error updating database schema: {e}")
        conn.rollback()
    finally:
        cur.close()
        conn.close()


# writing reply about stock request
def get_stock_info(symbol):
    fmp_data = get_fmp_data(symbol)
    intrinio_data = get_intrinio_data(symbol)

    response_text = f"Stock Information for {symbol}:\n\n"

    if fmp_data:
        response_text += f"1. Data from Financial Modeling Prep:\n"
        response_text += f"   Current Price: ${fmp_data['price']:.2f}\n"
        response_text += f"   Change: ${fmp_data['change']:.2f} ({fmp_data['changesPercentage']:.2f}%)\n"
    else:
        response_text += "Data from Financial Modeling Prep is not available.\n"

    if intrinio_data:
        response_text += f"\n2. Data from Intrinio:\n"
        response_text += f"   Last Price: ${intrinio_data['last_price']:.2f}\n"
        response_text += f"   Change: ${intrinio_data['change']:.2f} ({intrinio_data['change_percent']:.2f}%)\n"
    else:
        response_text += "\nData from Intrinio is not available.\n"

    return response_text


# obtained data from FMP
def get_fmp_data(symbol):
    url = f"https://financialmodelingprep.com/api/v3/quote/{symbol}?apikey={TOKEN_FMP}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        if data and isinstance(data, list) and len(data) > 0:
            return {
                'price': data[0]['price'],
                'change': data[0]['change'],
                'changesPercentage': data[0]['changesPercentage']
            }
        else:
            logger.warning(f"Unexpected response from FMP for {symbol}: {data}")
            return None
    except requests.RequestException as e:
        logger.error(f"Error fetching data from FMP for {symbol}: {e}")
        return None
    except (KeyError, ValueError, IndexError) as e:
        logger.error(f"Error parsing FMP data for {symbol}: {e}")
        return None


# obtained data from Intrinio ---------------------------------------------------------------------
def get_intrinio_data(symbol):
    url = f"https://api-v2.intrinio.com/securities/{symbol}/prices/realtime"
    params = {'api_key': TOKEN_INTRINIO}
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        if 'last_price' in data:
            return {
                'last_price': data['last_price'],
                'change': data['change'],
                'change_percent': data['change_percent']
            }
        else:
            logger.warning(f"Unexpected response from Intrinio for {symbol}: {data}")
            return None
    except requests.RequestException as e:
        logger.error(f"Error fetching data from Intrinio for {symbol}: {e}")
        if hasattr(e, 'response') and e.response is not None:
            logger.error(f"Response status code: {e.response.status_code}")
            logger.error(f"Response content: {e.response.text}")
        return None
    except (KeyError, ValueError) as e:
        logger.error(f"Error parsing Intrinio data for {symbol}: {e}")
        return None


# starting with bot (twice written)  --------------------------------------------------------------
if __name__ == '__main__':
    print("Bot is starting...")
    try:
        create_current_price_bot_table()
        update_database_schema()
        print("Database table created and schema updated successfully.")
    except Exception as e:
        print(f"Error setting up database: {e}")
        exit(1)

    print("Bot is polling...")
    bot.polling(none_stop=True, timeout=20000)  # wait for 5 hours
