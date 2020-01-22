from .atari import AtariDeepMindWrapper


env_wrapper_hub = {
    'atari_deepmind': AtariDeepMindWrapper,
    'identity': lambda x: x,
}
