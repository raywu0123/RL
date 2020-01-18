from abc import ABC, abstractmethod


class BaseScheduler(ABC):

    @abstractmethod
    def step(self):
        pass

    @abstractmethod
    def get_epsilon(self):
        pass
