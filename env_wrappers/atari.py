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
from .time_limit import TimeLimitWrapper


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


def make_atari(env_id):
    env = gym.make(env_id)
    assert 'NoFrameskip' in env.spec.id
    env = NoopResetWrapper(env, noop_max=30)
    env = MaxAndSkipWrapper(env, skip=4)
    return env


def wrap_deepmind(
        env,
        episode_life=True,
        clip_rewards=True,
        frame_stack=False,
        scale=False,
        max_frames=18000,
):
    """Configure environment for DeepMind-style Atari.
    """
    if episode_life:
        env = EpisodicLifeWrapper(env)
    if 'FIRE' in env.unwrapped.get_action_meanings():
        env = FireResetWrapper(env)
    env = WarpFrameWrapper(env)
    if scale:
        env = ScaledFloatFrame(env)
    if clip_rewards:
        env = ClipRewardWrapper(env)
    if frame_stack:
        env = StackFrameWrapper(env, 4)
    env = TimeLimitWrapper(env, max_frames)
    return env
