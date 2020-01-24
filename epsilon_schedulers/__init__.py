from .linear import LinearScheduler
from .exponential import ExponentialScheduler

scheduler_hub = {
    'linear': LinearScheduler,
    'exponential': ExponentialScheduler,
}
