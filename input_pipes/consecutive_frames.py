from collections import deque
from copy import deepcopy

import numpy as np

from .base import BaseInputPipe


class ConsecutiveFramesPipe(BaseInputPipe):

    def __init__(self, original_state_size: tuple, n: int):
        super().__init__(original_state_size)
        self.n = n
        self.state_queue = deque(maxlen=n)

    def reset(self):
        self.state_queue.clear()

    def __call__(self, state, **kwargs):
        if len(self.state_queue) == 0:
            for _ in range(self.n):
                self.state_queue.append(np.zeros_like(state))

        self.state_queue.append(state)
        consec_state = self.get_consec_state(self.state_queue)
        return consec_state

    @staticmethod
    def get_consec_state(state_queue: deque):
        frames = np.concatenate(state_queue, axis=-1)  # (H, W, C*n)
        return np.rollaxis(frames, axis=-1, start=0)

    def get_state_size(self) -> tuple:
        new_state_size = list(deepcopy(self.original_state_size))
        new_state_size[-1] *= self.n
        return tuple(new_state_size)
