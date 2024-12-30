import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(QNetwork, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 8),
            nn.ReLU(),
            nn.Linear(8, output_dim)
        )

    def forward(self, x):
        return self.layers(x)

class TradingAgent:
    reward_history = []

    def __init__(self, input_dim, evaluation_mode=False, model_file=""):
        self.input_dim = input_dim
        self.output_dim = 3  # sit, buy, sell
        self.memory_buffer = deque(maxlen=1000)
        self.holdings = []
        self.model_file = model_file
        self.evaluation_mode = evaluation_mode

        self.discount_factor = 0.95
        self.exploration_rate = 1.0
        self.exploration_min = 0.01
        self.exploration_decay = 0.995
        self.learning_rate = 0.001

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if evaluation_mode:
            self.model = torch.load(f"models/{model_file}")
        else:
            self.model = QNetwork(input_dim, self.output_dim).to(self.device)
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
            self.loss_function = nn.MSELoss()

    def select_action(self, state):
        if not self.evaluation_mode and random.random() <= self.exploration_rate:
            return random.randrange(self.output_dim)

        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).to(self.device)
            action_values = self.model(state_tensor)
            return action_values.argmax().item()

    def record_reward(self, reward):
        self.reward_history.append(reward)

    def replay_experience(self, batch_size):
        mini_batch = list(self.memory_buffer)[-batch_size:]

        for state, action, reward, next_state, done in mini_batch:
            target = reward

            if not done:
                state_tensor = torch.FloatTensor(state).to(self.device)
                next_state_tensor = torch.FloatTensor(next_state).to(self.device)

                with torch.no_grad():
                    target = reward + self.discount_factor * self.model(next_state_tensor).max(1)[0].item()

            state_tensor = torch.FloatTensor(state).to(self.device)
            target_values = self.model(state_tensor)
            target_values[0][action] = target

            self.optimizer.zero_grad()
            loss = self.loss_function(self.model(state_tensor), target_values)
            loss.backward()
            self.optimizer.step()

        if self.exploration_rate > self.exploration_min:
            self.exploration_rate *= self.exploration_decay

    def get_positive_rewards(self):
        positive_rewards = []
        for state, action, reward, next_state, done in self.memory_buffer:
            if reward > 0:
                positive_rewards.append(reward)
        return positive_rewards

    def get_all_rewards(self):
        return self.reward_history

# Helper functions remain mostly the same
def format_currency(value):
    return ("-$" if value < 0 else "$") + "{0:.2f}".format(abs(value))

def load_stock_data(symbol):
    data = []
    lines = open("C:/Users/pc/PythonDevEnv/MasterBigData&IoT/ReinforcementLearning/basicTrading/data/" + symbol + ".csv", "r").read().splitlines()
    for line in lines[1:]:
        data.append(float(line.split(",")[4]))
    return data

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def get_state_vector(data, t, window_size):
    start = t - window_size + 1
    block = data[start:t + 1] if start >= 0 else -start * [data[0]] + data[0:t + 1]
    state = []
    for i in range(window_size - 1):
        state.append(sigmoid(block[i + 1] - block[i]))
    return np.array([state])

# Training loop
if __name__ == "__main__":
    stock_symbol, window_size, num_episodes = 'GOLD', 3, 10
    agent = TradingAgent(window_size)
    stock_data = load_stock_data(stock_symbol)
    data_length = len(stock_data) - 1
    batch_size = 32

    total_profits = []
    buy_prices = []
    sell_prices = []
    transaction_log = []

    for episode in range(num_episodes + 1):
        print(f"Episode {episode}/{num_episodes}")
        state = get_state_vector(stock_data, 0, window_size + 1)
        total_profit = 0
        agent.holdings = []

        for t in range(data_length):
            action = agent.select_action(state)
            next_state = get_state_vector(stock_data, t + 1, window_size + 1)
            reward = 0

            if action == 1:  # buy
                agent.holdings.append(stock_data[t])
                print(f"Buy: {format_currency(stock_data[t])}")
                buy_prices.append(stock_data[t])
                transaction_log.append(f"{stock_data[t]}, Buy")

            elif action == 2 and len(agent.holdings) > 0:  # sell
                purchase_price = agent.holdings.pop(0)
                reward = max(stock_data[t] - purchase_price, 0)
                total_profit += stock_data[t] - purchase_price
                print(f"Sell: {format_currency(stock_data[t])} | Profit: {format_currency(stock_data[t] - purchase_price)}")

                total_profits.append(stock_data[t] - purchase_price)
                step_profit = stock_data[t] - purchase_price
                sell_prices.append(f"{stock_data[t]},{step_profit},{reward}")
                transaction_log.append(f"{stock_data[t]}, Sell")

            done = t == data_length - 1
            agent.memory_buffer.append((state, action, reward, next_state, done))
            state = next_state

            if done:
                print("--------------------------------")
                print(f"Total Profit: {format_currency(total_profit)}")
                print("--------------------------------")

            if len(agent.memory_buffer) > batch_size:
                agent.replay_experience(batch_size)

        if episode % 10 == 0:
            torch.save(agent.model, f"models/model_ep{episode}")