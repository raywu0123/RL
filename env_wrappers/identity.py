import gym


class IdentityWrapper(gym.Wrapper):

    def __init__(self, env, **kwargs):
        super().__init__(env)
