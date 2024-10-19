import numpy as np
import random



class TradingEnvironment:
    def __init__(self, data):
        self.data = data
        self.current_step = 0
        self.max_steps = len(data) - 1
        self.initial_balance = 10000
        self.balance = self.initial_balance
        self.holdings = 0

    def reset(self):
        self.current_step = 0
        self.balance = self.initial_balance
        self.holdings = 0
        return self._get_state()

    def step(self, action):
        # 0: Hold, 1: Buy, 2: Sell
        current_price = self.data[self.current_step]
        reward = 0

        if action == 1 and self.balance >= current_price:  # Buy
            self.holdings += 1
            self.balance -= current_price
            reward = -current_price
        elif action == 2 and self.holdings > 0:  # Sell
            self.holdings -= 1
            self.balance += current_price
            reward = current_price

        self.current_step += 1
        done = self.current_step >= self.max_steps
        next_state = self._get_state()
        
        return next_state, reward, done

    def _get_state(self):
        return [self.balance, self.holdings, self.data[self.current_step]]

class QLearningAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.q_table = {}
        self.alpha = 0.1  # Learning rate
        self.gamma = 0.95  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01

    def get_action(self, state):
        state_key = tuple(state)
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.action_size)

        if random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)
        return np.argmax(self.q_table[state_key])

    def update(self, state, action, reward, next_state, done):
        state_key = tuple(state)
        next_state_key = tuple(next_state)

        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.action_size)
        if next_state_key not in self.q_table:
            self.q_table[next_state_key] = np.zeros(self.action_size)

        current_q = self.q_table[state_key][action]
        if done:
            max_future_q = 0
        else:
            max_future_q = np.max(self.q_table[next_state_key])
        
        new_q = current_q + self.alpha * (reward + self.gamma * max_future_q - current_q)
        self.q_table[state_key][action] = new_q

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

def train_agent(env, agent, episodes):
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            action = agent.get_action(state)
            next_state, reward, done = env.step(action)
            agent.update(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

        if episode % 100 == 0:
            print(f"Episode: {episode}, Total Reward: {total_reward}, Epsilon: {agent.epsilon:.2f}")

# Example usage
if __name__ == "__main__":
    # Generate some dummy price data
    price_data = np.random.randint(100, 200, size=1000)
    
    env = TradingEnvironment(price_data)
    agent = QLearningAgent(state_size=3, action_size=3)
    
    train_agent(env, agent, episodes=1000)

    # Test the trained agent
    state = env.reset()
    total_reward = 0
    done = False

    while not done:
        action = agent.get_action(state)
        next_state, reward, done = env.step(action)
        state = next_state
        total_reward += reward

    print(f"Final balance: {env.balance:.2f}")
    print(f"Final holdings: {env.holdings}")
    print(f"Total reward: {total_reward:.2f}")

