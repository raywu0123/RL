from .base import BaseInputPipe


class IdentityPipe(BaseInputPipe):

    def __init__(self, original_shape):
        super().__init__(original_shape)

    @staticmethod
    def __call__(state):
        return state
