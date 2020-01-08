from functools import partial

from .fc import FCNetwork

network_hub = {
    'fc24': partial(FCNetwork, n_hidden=24),
}
