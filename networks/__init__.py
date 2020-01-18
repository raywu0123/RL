from functools import partial

from .fc import FCNetwork
from .cnn import CNNNetwork

network_hub = {
    'fc24': partial(FCNetwork, n_hidden=24),
    'fc256': partial(FCNetwork, n_hidden=256),
    'cnn': CNNNetwork,
}
