import torch
from torch import nn


class CNNNetwork(nn.Module):

    def __init__(self, state_size, action_space):
        super().__init__()
        self.state_size = state_size
        input_chns = state_size[-1]
        self.convs = nn.Sequential(
            nn.Conv2d(input_chns, 16, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2),
            nn.ReLU(),
        )
        self.dense = nn.Sequential(
            nn.Linear(self.get_dense_input_dim(state_size, self.convs), 256),
            nn.ReLU(),
            nn.Linear(256, action_space.n),
        )

    @staticmethod
    def get_dense_input_dim(state_size, convs):
        mock_input = torch.empty([1, *state_size]).permute([0, 3, 1, 2])
        output = convs(mock_input).view(1, -1)
        return output.shape[-1]

    def forward(self, state: torch.Tensor):
        if not state.ndim == 4:
            raise ValueError(f'Invalid shape, got {state.shape}')
        state = state.permute([0, 3, 1, 2])
        x = self.convs(state)   # (N, C, H, W)
        x = x.view(x.shape[0], -1)
        x = self.dense(x)
        return x
