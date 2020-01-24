import gym
import numpy as np


class ScaledFloatFrame(gym.ObservationWrapper):

    def observation(self, observation):
        # careful! This undoes the memory optimization, use
        # with smaller replay buffers only.
        return np.array(observation).astype(np.float32) / 255.0
