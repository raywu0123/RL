from abc import ABC, abstractmethod


class BaseInputPipe(ABC):

    def __init__(self, original_state_size):
        self.original_state_size = original_state_size

    @staticmethod
    @abstractmethod
    def __call__(state):
        pass

    def reset(self):
        pass

    def get_state_size(self):
        return self.original_state_size
