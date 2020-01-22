from torch import nn


class FCNetwork(nn.Module):

    def __init__(self, state_size, action_space, n_hidden):
        super().__init__()
        self.state_size = state_size
        self.dense = nn.Sequential(
            nn.Linear(state_size[0], n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, action_space.n),
        )

    def forward(self, state):
        return self.dense(state)
