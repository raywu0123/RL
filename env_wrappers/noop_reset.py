import gym
import numpy as np


class NoopResetWrapper(gym.Wrapper):

    def __init__(self, env, noop_max=30):
        """Sample initial states by taking random number of no-ops on reset.
        No-op is assumed to be action 0.
        """
        super().__init__(env)
        self.noop_max = noop_max
        self.override_num_noops = None
        if isinstance(env.action_space, gym.spaces.MultiBinary):
            self.noop_action = np.zeros(self.env.action_space.n, dtype=np.int64)
        else:
            # used for atari environments
            self.noop_action = 0
            assert env.unwrapped.get_action_meanings()[0] == 'NOOP'

    def _reset(self, **kwargs):
        """ Do no-op action for a number of steps in [1, noop_max]."""
        self.env.reset(**kwargs)
        if self.override_num_noops is not None:
            noops = self.override_num_noops
        else:
            noops = self.unwrapped.np_random.randint(1, self.noop_max + 1)  # pylint: disable=E1101
        assert noops > 0
        obs = None
        for _ in range(noops):
            obs, _, done, _ = self.env.step(self.noop_action)
            if done:
                obs = self.env.reset(**kwargs)
        return obs
