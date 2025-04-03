import torch
import numpy as np
from agents.ddpg_agent import DDPGAgent
from envs.trading_env import TradingEnv

data = np.load("data.npy")
env = TradingEnv(data)
state_dim = env.reset().shape[0]
action_dim = 1
max_action = 1.0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
agent = DDPGAgent(state_dim, action_dim, max_action, device)

episodes = 100
for ep in range(episodes):
    state = env.reset()
    episode_reward = 0
    done = False
    while not done:
        action = agent.select_action(state)
        next_state, reward, done, _ = env.step(action)
        agent.replay_buffer.add((state, action, reward, next_state, float(done)))
        state = next_state
        episode_reward += reward
        agent.train(batch_size=64)
    print(f"Episode {ep + 1}, Reward: {episode_reward}")