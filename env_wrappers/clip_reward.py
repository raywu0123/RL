import gym
import numpy as np


class ClipRewardWrapper(gym.RewardWrapper):

    def reward(self, reward):
        """Bin reward to {+1, 0, -1} by its sign."""
        return np.sign(reward)
