from torch import nn


class FutureNetwork(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, latent_state):
        pass
        # return reality_fn_params, p_action


class RealityFunction(nn.Module):

    def __init__(self, parameters):
        super().__init__()

    def forward(self, t):
        pass
