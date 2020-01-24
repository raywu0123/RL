from abc import ABC, abstractmethod


class BaseScheduler(ABC):

    @abstractmethod
    def step(self, num=1):
        pass

    @abstractmethod
    def get_epsilon(self):
        pass
