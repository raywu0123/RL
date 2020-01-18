import torch
from torch import nn


class CNNNetwork(nn.Module):

    def __init__(self, state_size, action_space):
        super().__init__()
        self.state_size = state_size
        input_chns = state_size[-1]
        self.convs = nn.Sequential(
            nn.BatchNorm2d(num_features=input_chns),

            nn.Conv2d(input_chns, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=8),

            nn.Conv2d(8, 4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=4),

            nn.Conv2d(4, 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=2),

            nn.Conv2d(2, action_space.n, kernel_size=3, padding=1),
        )

    def forward(self, state: torch.Tensor):
        if not state.ndim == 4:
            raise ValueError(f'Invalid shape, got {state.shape}')
        x = self.convs(state)   # (N, C, H, W)
        x = x.view(x.shape[0], x.shape[1], -1)
        return x.mean(dim=-1)
