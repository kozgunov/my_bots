import numpy as np
import time

class TradingEnvironment:
    def __init__(self, data, initial_balance=100000.0, fee=0.001, stop_loss=0.05, risk_per_trade=0.02):
        self.data = data
        self.initial_balance = initial_balance
        self.fee = fee
        self.stop_loss = stop_loss
        self.risk_per_trade = risk_per_trade
        self.reset()

    def reset(self):
        self.current_step = 0
        self.balance = self.initial_balance
        self.position = 0
        self.entry_price = 0
        self.total_trades = 0
        self.winning_trades = 0
        self.total_profit = 0
        return self._get_state()

    def step(self, action):
        current_price = self.data[self.current_step]['close']
        volume = self.data[self.current_step]['volume']
        reward = 0
        done = False

        # Apply stop loss
        if self.position != 0 and abs(current_price - self.entry_price) / self.entry_price >= self.stop_loss:
            reward = self._close_position(current_price)
            done = True

        # Action: 0 (hold), 1 (buy), 2 (sell)
        elif action == 1:  # Buy
            if self.position <= 0:
                position_size = self.balance * self.risk_per_trade / current_price
                cost = position_size * current_price * (1 + self.fee)
                if cost <= self.balance:
                    self.position += position_size
                    self.balance -= cost
                    self.entry_price = current_price
                    self.total_trades += 1

        elif action == 2:  # Sell
            if self.position >= 0:
                reward = self._close_position(current_price)

        # Modify the reward calculation
        if action != 0:  # If not holding
            self.total_trades += 1
            if reward > 0:
                self.winning_trades += 1
            
            # Increase reward for making trades
            reward += 0.1  # Small bonus for taking action
            
            # Adjust reward based on win rate
            current_win_rate = self.winning_trades / self.total_trades if self.total_trades > 0 else 0
            reward *= (1 + current_win_rate)  # Amplify reward for higher win rates

        self.current_step += 1
        if self.current_step >= len(self.data) - 1:
            done = True

        next_state = self._get_state()
        info = {
            'balance': self.balance,
            'position': self.position,
            'entry_price': self.entry_price,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'total_profit': self.total_profit
        }
        
        return next_state, reward, done, info

    def _close_position(self, current_price):
        gain = self.position * (current_price - self.entry_price)
        self.balance += self.position * current_price * (1 - self.fee) + gain
        reward = gain / self.initial_balance  # Normalize reward
        self.total_profit += gain
        if gain > 0:
            self.winning_trades += 1
        self.position = 0
        self.entry_price = 0
        return reward

    def _get_state(self):
        current_price = self.data[self.current_step]['close']
        state = np.array([
            self.balance / self.initial_balance,
            self.position * current_price / self.initial_balance,
            current_price / self.data[0]['close'],
            self.data[self.current_step]['volume'] / np.mean([d['volume'] for d in self.data]),
            (current_price - self.entry_price) / self.entry_price if self.position != 0 else 0,
            self.total_trades / 1000,  # Normalize total trades
            self.winning_trades / max(1, self.total_trades),  # Current win rate
        ])
        return state.flatten()
