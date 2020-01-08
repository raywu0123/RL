from abc import ABC, abstractmethod

from gym import spaces


class BaseAgent(ABC):

    def __init__(self, state_size, action_space: spaces.Space, **kwargs):
        self.state_size = state_size
        self.action_space = action_space

    @abstractmethod
    def end_timestep(self, state, next_state, action, done, info, reward):
        pass

    @abstractmethod
    def end_episode(self):
        pass

    def get_random_action(self):
        return self.action_space.sample()

    @abstractmethod
    def get_action(self, state):
        pass