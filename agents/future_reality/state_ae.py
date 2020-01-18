from math import ceil, floor

from torch import nn
from torch.nn import functional as F


class StateEncoder(nn.Module):

    def __init__(self, state_size):
        super().__init__()
        self.state_size = state_size
        input_chns = state_size[-1]
        self.convs = nn.Sequential(
            nn.BatchNorm2d(num_features=input_chns),

            nn.Conv2d(input_chns, 32, kernel_size=8, stride=4),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(num_features=32),

            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(num_features=64),

            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(num_features=64),
        )

    def forward(self, state):
        return self.convs(state)


class StateDecoder(nn.Module):

    def __init__(self, state_size):
        super().__init__()
        self.state_size = state_size
        output_chns = state_size[-1]
        self.deconvs = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(num_features=64),

            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(num_features=32),

            nn.ConvTranspose2d(32, output_chns, kernel_size=8, stride=4),
        )

    def forward(self, latent_state):
        out = self.deconvs(latent_state)
        h_diff = self.state_size[0] - out.shape[-2]
        w_diff = self.state_size[1] - out.shape[-1]
        out = F.pad(
            out,
            (floor(w_diff / 2), ceil(w_diff / 2), floor(h_diff / 2), ceil(h_diff / 2)),
        )
        return out
