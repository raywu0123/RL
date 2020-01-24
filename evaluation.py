import gym
import numpy as np
import torch

from agents import BaseAgent


def evaluate_agent(agent: BaseAgent, env: gym.Env, n_episodes: int = 100) -> dict:
    agent.eval()
    episode_rewards = []
    for i_episode in range(n_episodes):
        state = torch.from_numpy(env.reset()).float()
        t = 0
        done = False
        rewards = []
        while not done:
            action = agent.get_action(state)
            next_state, reward, done, info = env.step(action)
            next_state = torch.from_numpy(next_state).float()
            rewards.append(reward)
            state = next_state
            t += 1

        episode_reward = np.sum(rewards)
        episode_rewards.append(episode_reward)
        _ = agent.end_episode()

    agent.train()
    return {
        'eval_mean_reward': np.mean(episode_rewards)
    }
