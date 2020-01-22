from functools import partial

from .atari import AtariDeepMindWrapper
from .identity import IdentityWrapper


class EnvWrapperHub:

    wrappers = {
        'atari_deepmind': AtariDeepMindWrapper,
        'identity': IdentityWrapper,
    }

    @classmethod
    def get_wrapper(cls, wrapper_id, is_train: bool):
        return partial(
            cls.wrappers[wrapper_id],
            is_train=is_train,
        )
