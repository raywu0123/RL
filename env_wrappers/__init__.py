from .atari_wrapper import make_wrap_atari

env_wrapper_hub = {
    'atari': make_wrap_atari,
    'identity': lambda x: x,
}
