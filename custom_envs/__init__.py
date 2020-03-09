import gym
from argparse import Namespace

from .text_puzzle import TextPuzzle


class EnvironmentFactory:

    custom_envs = {
        'text_puzzle': TextPuzzle
    }

    @classmethod
    def make(cls, key: str, args: Namespace):
        if key in cls.custom_envs:
            return cls.custom_envs[key](args)
        else:
            return gym.make(key)
