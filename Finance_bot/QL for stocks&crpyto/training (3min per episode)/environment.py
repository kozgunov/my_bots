import numpy as np
import logging
from utils import calculate_rsi, calculate_macd, calculate_bollinger_bands

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class TradingEnvironment:
    def __init__(self, data, window_size=30, initial_balance=100000.0, fee=0.0015, stop_loss=0.05, risk_per_trade=0.02, 
                 position_limit=0.5, volatility_factor=0.001, risk_free_rate=0.02/365, trade_fraction=0.2):
        logging.info(f"Initializing TradingEnvironment with {len(data)} data points")
        self.data = data
        self.window_size = window_size
        self.initial_balance = initial_balance
        self.fee = fee
        self.stop_loss = stop_loss
        self.risk_per_trade = risk_per_trade
        self.position_limit = position_limit
        self.volatility_factor = volatility_factor
        self.risk_free_rate = risk_free_rate
        self.trade_fraction = trade_fraction  # New parameter
        self.volume_ma = self.calculate_volume_ma(data)
        logging.debug(f"Calculated volume MA with shape {self.volume_ma.shape}")
        self.reset()
        self.negative_fee_factor = 0.0001  # Small fee factor for negative rewards

    def calculate_volume_ma(self, data, window=20):
        try:
            volumes = [d['volume'] for d in data]
            return np.convolve(volumes, np.ones(window), 'valid') / window
        except KeyError:
            print("Warning: 'volume' key not found in data. Using default values.")
            return np.ones(len(data))  # return array of ones as fallback

    def reset(self):
        self.current_step = self.window_size
        self.balance = self.initial_balance
        self.position = 0
        self.entry_price = 0
        self.total_trades = 0
        self.winning_trades = 0
        self.total_profit = 0
        self.risk_free_asset = 0
        return self._get_state()

    def step(self, action):
        try:
            logging.debug(f"Step: current_step={self.current_step}, action={action}")
            
            if self.current_step >= len(self.data):
                logging.error(f"current_step ({self.current_step}) is out of bounds for data length ({len(self.data)})")
                return self._get_state(), 0, True, {}

            current_data = self.data[self.current_step]
            logging.debug(f"Current data: {current_data}")
            
            current_price = current_data['close']
            volume = current_data['volume']
            reward = 0
            done = False
            trade_made = False
            trade_amount = 0

            # stop loss
            if self.position != 0 and abs(current_price - self.entry_price) / self.entry_price >= self.stop_loss:
                reward = self._close_position(current_price)
                done = True

            # actions: 0 (hold), 1 (buy), 2 (sell), 3 (risk-free asset)
            elif action == 1:  # buy
                if self.balance > 0:
                    max_trade_size = min(self.balance * self.trade_fraction, self.initial_balance * self.position_limit)
                    trade_size = max_trade_size  # The agent decides to use the maximum allowed trade size
                    price_impact = self._calculate_price_impact(trade_size, volume, current_price)
                    execution_price = current_price * (1 + price_impact)
                    shares_bought = trade_size / execution_price
                    self.position += shares_bought
                    self.balance -= trade_size * (1 + self.fee)
                    self.entry_price = (self.entry_price * (self.position - shares_bought) + execution_price * shares_bought) / self.position
                    trade_made = True
                    trade_amount = trade_size
            elif action == 2:  # sell
                if self.position > 0:
                    max_sell_fraction = min(self.trade_fraction, 1.0)  # Can't sell more than 100% of position
                    shares_to_sell = self.position * max_sell_fraction
                    trade_size = shares_to_sell * current_price
                    price_impact = self._calculate_price_impact(trade_size, volume, current_price)
                    execution_price = current_price * (1 - price_impact)
                    self.balance += shares_to_sell * execution_price * (1 - self.fee)
                    self.position -= shares_to_sell
                    reward = (execution_price - self.entry_price) * shares_to_sell / self.initial_balance
                    if self.position == 0:
                        self.entry_price = 0
                    trade_made = True
                    trade_amount = trade_size
            elif action == 3:  # move to risk-free asset
                amount_to_move = self.balance * self.trade_fraction
                self.risk_free_asset += amount_to_move
                self.balance -= amount_to_move

            # apply market volatility
            market_move = np.random.normal(0, self.volatility_factor)
            current_price *= (1 + market_move)

            # Update risk-free asset
            self.risk_free_asset *= (1 + self.risk_free_rate)

            # modify the reward calculation
            if action != 0:  # if not holding
                self.total_trades += 1
                if reward > 0:
                    self.winning_trades += 1
                
                # increase reward for making trades
                reward += 0.1  # Small bonus for taking action
                
                # adjust reward based on win rate
                current_win_rate = self.winning_trades / self.total_trades if self.total_trades > 0 else 0
                reward *= (1 + current_win_rate)  # Amplify reward for higher win rates

            # negative reward fee
            if reward < 0:
                fee = abs(reward) * self.negative_fee_factor
                reward -= fee

            # negative balance fee
            if self.balance < 0:
                fee = abs(self.balance) * self.negative_fee_factor
                reward -= fee

            self.current_step += 1
            if self.current_step >= len(self.data) - 1:
                done = True

            next_state = self._get_state()
            info = {
                'trade_made': trade_made,
                'total_trades': self.total_trades,
                'winning_trades': self.winning_trades,
                'current_price': current_price,
                'balance': self.balance,
                'position': self.position,
                'risk_free_asset': self.risk_free_asset,
                'current_volume': volume,
                'trade_amount': trade_amount
            }
            
            logging.debug(f"Step completed: reward={reward}, done={done}")
            return next_state, reward, done, info
        except Exception as e:
            logging.error(f"Error in step method: {e}", exc_info=True)
            raise

    def _close_position(self, current_price):
        trade_size = self.position * current_price
        price_impact = self._calculate_price_impact(trade_size, self.data[self.current_step]['volume'], current_price)
        execution_price = current_price * (1 - price_impact)
        gain = self.position * (execution_price - self.entry_price)
        self.balance += self.position * execution_price * (1 - self.fee) + gain
        reward = gain / self.initial_balance  # Normalize reward
        self.total_profit += gain
        if gain > 0:
            self.winning_trades += 1
        self.position = 0
        self.entry_price = 0
        return reward

    def _get_state(self):
        try:
            start = max(0, self.current_step - self.window_size)
            end = self.current_step
            window = self.data[start:end]
            
            state = np.array([
                [d['open'], d['high'], d['low'], d['close'], d['volume']] for d in window
            ])
            
            if self.current_step < len(self.volume_ma):
                volume_ma = self.volume_ma[self.current_step]
            else:
                volume_ma = self.volume_ma[-1]
            
            current_volume = self.data[self.current_step]['volume']
            volume_ratio = current_volume / volume_ma if volume_ma > 0 else 1

            # more technical indicators
            current_price = self.data[self.current_step]['close']
            sma_20 = np.mean([d['close'] for d in self.data[max(0, self.current_step-20):self.current_step]])
            sma_50 = np.mean([d['close'] for d in self.data[max(0, self.current_step-50):self.current_step]])
            rsi = calculate_rsi([d['close'] for d in self.data[max(0, self.current_step-14):self.current_step+1]])[-1]
            macd, signal = calculate_macd([d['close'] for d in self.data[max(0, self.current_step-26):self.current_step+1]])
            _, upper, lower = calculate_bollinger_bands([d['close'] for d in self.data[max(0, self.current_step-20):self.current_step+1]])

            additional_info = np.array([
                self.balance / self.initial_balance,
                self.position * current_price / self.initial_balance,
                self.total_trades / 1000,
                self.winning_trades / max(1, self.total_trades),
                self.risk_free_asset / self.initial_balance,
                volume_ratio,
                (current_price - sma_20) / sma_20,  # price relative to SMA20
                (current_price - sma_50) / sma_50,  # price relative to SMA50
                rsi / 100,  # normalized RSI
                (macd[-1] - signal[-1]) / current_price,  # MACD histogram
                (current_price - lower[-1]) / (upper[-1] - lower[-1]),  # bollinger Band position
                self.entry_price / current_price if self.position != 0 else 1,  # current position's performance
            ])
            
            return np.concatenate((state.flatten(), additional_info))
        except Exception as e:
            logging.error(f"Error in _get_state method: {e}")
            raise

    def _calculate_price_impact(self, trade_size, volume, price):
        # square-root model for price impact
        return 0.1 * np.sqrt(trade_size / (volume * price))
