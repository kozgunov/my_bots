import threading
from environment import TradingEnvironment
from agent import ImprovedQLearningAgent
from utils import prepare_data, create_sequences, get_crypto_info, get_latest_news
from config import *
import telebot
from telebot.types import ReplyKeyboardMarkup, KeyboardButton
import matplotlib.pyplot as plt
import io
import numpy as np
import os
import logging

# setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

bot = telebot.TeleBot(TOKEN_TG)

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
    keyboard.row(
        KeyboardButton("Predict Crypto Market"),
        KeyboardButton("Free Subscription")
    )
    keyboard.row(KeyboardButton("Non-free Subscription"))
    return keyboard

@bot.message_handler(commands=['start', 'help'])
def send_welcome(message):
    try:
        help_text = (
            "Welcome to the Crypto Trading Bot! Here are the commands you can use:\n"
            "/help - Show this help message\n"
            "/predict - Predict crypto market\n"
            "/free_subscription - Get free subscription\n"
            "/non_free_subscription - Get non-free subscription\n"
        )
        bot.send_message(message.chat.id, help_text, reply_markup=create_main_keyboard())
    except Exception as e:
        logger.error(f"Error in send_welcome: {e}")
        bot.send_message(message.chat.id, "An error occurred. Please try again later.")

@bot.message_handler(func=lambda message: message.text == "Predict Crypto Market")
@bot.message_handler(commands=['predict'])
def predict_crypto_market(message):
    try:
        bot.send_message(message.chat.id, "Preparing to make a prediction. This may take a moment...")

        # load&prepare data
        df, scaler = prepare_data('binance', SYMBOL, timeframe=TIMEFRAME, limit=DATA_LIMIT)
        
        if df is None:
            bot.send_message(message.chat.id, "Failed to load data. Please try again later.")
            return

        # create sequences
        seq_length = 24  # 24 hours of data
        X, y = create_sequences(df, seq_length)
        y = y[:, df.columns.get_loc('close')]

        # use the last sequence for prediction
        last_sequence = X[-1:]

        # create environment and agent
        env = TradingEnvironment(y[-60:], initial_balance=INITIAL_BALANCE, fee=TRADING_FEE)  # Use last 60 points for visualization
        agent = ImprovedQLearningAgent(state_size=X.shape[1] * X.shape[2] + 2, action_size=3)

        # load the trained model
        if os.path.exists("best_model.pth"):
            agent.load("best_model.pth")
        else:
            bot.send_message(message.chat.id, "Trained model not found. Please train the model first.")
            return

        # make prediction
        state = env.reset()
        action = agent.get_action(last_sequence)
        prediction = "Buy" if action == 1 else "Sell" if action == 2 else "Hold"

        # 1-hour graph of Bitcoin
        plt.figure(figsize=(10, 6))
        plt.plot(df.index[-60:], df['close'][-60:])
        plt.title("Bitcoin Price (Last 60 Hours)")
        plt.xlabel("Time")
        plt.ylabel("Price")
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png')
        img_buffer.seek(0)
        
        # send the plot
        bot.send_photo(message.chat.id, img_buffer)

        # send prediction
        bot.send_message(message.chat.id, f"Prediction: {prediction}")

        # latest news
        news = get_latest_news("Bitcoin")
        bot.send_message(message.chat.id, news, parse_mode='Markdown')

        # general indicator data
        crypto_info = get_crypto_info("BTC")
        bot.send_message(message.chat.id, crypto_info, parse_mode='Markdown')

    except Exception as e:
        logger.error(f"Error in predict_crypto_market: {e}")
        bot.send_message(message.chat.id, "An error occurred while making the prediction. Please try again later.")

@bot.message_handler(func=lambda message: message.text == "Free Subscription")
@bot.message_handler(commands=['free_subscription'])
def free_subscription(message):
    try:
        bot.send_message(message.chat.id, "Hello, world!")
    except Exception as e:
        logger.error(f"Error in free_subscription: {e}")
        bot.send_message(message.chat.id, "An error occurred. Please try again later.")

@bot.message_handler(func=lambda message: message.text == "Non-free Subscription")
@bot.message_handler(commands=['non_free_subscription'])
def non_free_subscription(message):
    try:
        bot.send_message(message.chat.id, "Hello, world!")
    except Exception as e:
        logger.error(f"Error in non_free_subscription: {e}")
        bot.send_message(message.chat.id, "An error occurred. Please try again later.")

@bot.message_handler(func=lambda message: True)
def handle_message(message):
    try:
        if message.text == "Show Current Stocks":
            bot.send_message(message.chat.id, "Current stock information is not available in this demo.")
        elif message.text == "Show Current Crypto":
            crypto_info = get_crypto_info("BTC")
            bot.send_message(message.chat.id, crypto_info, parse_mode='Markdown')
        elif message.text == "Get My Data":
            bot.send_message(message.chat.id, "User data retrieval is not implemented in this demo.")
        elif message.text == "Remove My Data":
            bot.send_message(message.chat.id, "User data removal is not implemented in this demo.")
        else:
            bot.send_message(message.chat.id, "I don't understand that command. Please use the keyboard or type /help for available commands.")
    except Exception as e:
        logger.error(f"Error in handle_message: {e}")
        bot.send_message(message.chat.id, "An error occurred. Please try again later.")

def start_bot():
    try:
        bot.polling(none_stop=True)
    except Exception as e:
        logger.error(f"Error in bot polling: {e}")

def main():
    try:
        # start the Telegram bot in a separate thread
        bot_thread = threading.Thread(target=start_bot)
        bot_thread.start()

        logger.info("Bot is running. Press CTRL+C to stop.")

        bot_thread.join()
    except KeyboardInterrupt:
        logger.info("Bot stopped by user.")
    except Exception as e:
        logger.error(f"Unexpected error in main: {e}")

if __name__ == "__main__":
    main()
