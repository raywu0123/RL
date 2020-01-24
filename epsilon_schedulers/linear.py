from .base import BaseScheduler


class LinearScheduler(BaseScheduler):

    def __init__(self, init_value=1., decay=0.05, min_value=0.01):
        self.min_value = min_value
        self.epsilon = init_value
        self.decay = decay

    def step(self, num=1):
        self.epsilon = max(self.min_value, self.epsilon - self.decay * num)

    def get_epsilon(self):
        return self.epsilon
