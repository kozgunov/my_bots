import numpy as np
import time

class TradingEnvironment:
    def __init__(self, data, initial_balance=100.0, fee=0.001, max_drawdown=0.6, episode_duration=45):
        self.data = data
        self.initial_balance = initial_balance
        self.fee = fee
        self.max_drawdown = max_drawdown
        self.episode_duration = episode_duration
        self.reset()

    def reset(self):
        self.current_step = 0
        self.balance = self.initial_balance
        self.holdings = 0
        self.last_action_time = 0
        self.max_balance = self.initial_balance
        self.start_time = time.time()
        return self._get_state()

    def step(self, action):
        current_price = self.data[self.current_step]
        reward = 0
        done = False

        # Implement a cooldown period between actions
        time_since_last_action = self.current_step - self.last_action_time

        if action == 1 and self.balance > 0 and self.holdings == 0 and time_since_last_action >= 10:  # Buy
            max_buy = min(self.balance / (current_price * (1 + self.fee)), self.balance * 0.2)  # Limit to 20% of balance
            self.holdings = max_buy
            self.balance -= max_buy * current_price * (1 + self.fee)
            self.last_action_time = self.current_step
        elif action == 2 and self.holdings > 0 and time_since_last_action >= 10:  # Sell
            sell_amount = self.holdings * current_price * (1 - self.fee)
            self.balance += sell_amount
            reward = sell_amount - (self.holdings * self.data[self.last_action_time] * (1 + self.fee))
            self.holdings = 0
            self.last_action_time = self.current_step

        self.current_step += 1
        if self.current_step >= len(self.data) - 1:
            done = True

        # Update max balance and check for max drawdown
        current_total = self.balance + self.holdings * current_price
        self.max_balance = max(self.max_balance, current_total)
        current_drawdown = (self.max_balance - current_total) / self.max_balance
        if current_drawdown > self.max_drawdown:
            done = True
            reward -= 10  # Penalty for exceeding max drawdown

        # Check if episode duration has been exceeded
        if time.time() - self.start_time > self.episode_duration:
            done = True

        # Implement a small negative reward for holding to encourage action
        if action == 0:
            reward -= 0.01

        next_state = self._get_state()
        
        return next_state, reward, done, {"early_termination": done and self.current_step < len(self.data) - 1}

    def _get_state(self):
        state = np.array([
            self.balance / self.initial_balance,
            self.holdings * self.data[self.current_step] / self.initial_balance,
            self.data[self.current_step] / self.data[0],
            (self.current_step - self.last_action_time) / 10  # Normalized time since last action
        ])
        return state
