import gym
import numpy as np

class TradingEnv(gym.Env):
    def __init__(self, data):
        self.data = data
        self.current_step = 0
        self.max_step = len(data) - 1
        self.balance = 1000
        self.position = 0

    def reset(self):
        self.current_step = 0
        self.balance = 1000
        self.position = 0
        return self._get_state()

    def _get_state(self):
        return self.data[self.current_step]

    def step(self, action):
        price = self.data[self.current_step][0]
        self.position += action[0]
        self.balance -= action[0] * price
        self.current_step += 1
        reward = self.balance + self.position * price
        done = self.current_step >= self.max_step
        return self._get_state(), reward, done, {}

    def render(self):
        pass