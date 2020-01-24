"""original code:
github.com/openai/baselines/blob/master/baselines/common/atari_wrappers.py
"""
import gym

from .noop_reset import NoopResetWrapper
from .max_and_skip import MaxAndSkipWrapper
from .episodic_life import EpisodicLifeWrapper
from .fire_reset import FireResetWrapper
from .stack_frame import StackFrameWrapper
from .warp_frame import WarpFrameWrapper
from .scale_float_frame import ScaledFloatFrame
from .clip_reward import ClipRewardWrapper


class AtariWrapper(gym.Wrapper):

    def __init__(self, env):
        if env.spec is not None:
            assert 'NoFrameskip' in env.spec.id
        env = NoopResetWrapper(env, noop_max=30)
        env = MaxAndSkipWrapper(env, skip=4)
        super().__init__(env)


class AtariDeepMindWrapper(gym.Wrapper):

    def __init__(self, env, is_train: bool):
        env = AtariWrapper(env)
        env = EpisodicLifeWrapper(env)
        if 'FIRE' in env.unwrapped.get_action_meanings():
            env = FireResetWrapper(env)
        env = WarpFrameWrapper(env)
        if is_train:
            env = ScaledFloatFrame(env)
        env = ClipRewardWrapper(env)
        env = StackFrameWrapper(env, 4)
        super().__init__(env)
