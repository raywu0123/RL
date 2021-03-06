from abc import ABC, abstractmethod

from gym import spaces


class BaseAgent(ABC):

    def __init__(
        self,
        action_space: spaces.Space,
        **kwargs,
    ):
        self.action_space = action_space

    @abstractmethod
    def end_timestep(self, info, **kwargs):
        pass

    @abstractmethod
    def end_episode(self) -> dict:
        pass

    def get_random_action(self):
        return self.action_space.sample()

    @abstractmethod
    def get_action(self, state):
        pass

    @abstractmethod
    def load(self, checkpoint_dir):
        pass

    @abstractmethod
    def eval(self):
        pass

    @abstractmethod
    def train(self):
        pass
