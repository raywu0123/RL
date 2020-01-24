import torch
import torch.nn.functional as F
import torch.optim as optim

from .base import BaseAgent
from networks import network_hub


class DQNAgent(BaseAgent):

    def __init__(
        self,
        state_size,
        device,
        network_id: str,
        history_length: int,
        batch_size: int = 32,
        gamma: float = 0.99,
        lr: float = 1e-4,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.history_length = history_length
        self.device = device
        self.batch_size = batch_size
        self.gamma = gamma

        # Q-Network
        self.qnetwork_local = network_hub[network_id](
            state_size,
            history_length,
            self.action_space,
        ).to(self.device)
        self.qnetwork_target = network_hub[network_id](
            state_size,
            history_length,
            self.action_space,
        ).to(self.device)
        self.qnetwork_target.eval()
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=lr)

        # Replay memory
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
        self.is_eval = False

    def learn(self, states, actions, rewards, next_states, dones):
        dones = dones.float()
        # Save experience in replay memory
        self.t_step += 1
        self.epsilon_scheduler.step()
        # Get max predicted Q values (for next states) from target model
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0]
        # Compute Q targets for current states
        Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))

        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions.view(-1, 1)).squeeze()

        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss

    def update_target_net(self):
        for target_param, local_param in zip(
                self.qnetwork_target.parameters(), self.qnetwork_local.parameters()
        ):
            target_param.data.copy_(local_param.data)

    def end_episode(self):
        return {
            'epsilon': self.epsilon_scheduler.get_epsilon(),
        }

    def act(self, state, epsilon=0.):
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()
        action_values = action_values.argmax(-1).to(state.device)
        mask = torch.rand(state.size(0), device=state.device, dtype=torch.float32) < epsilon
        masked = mask.sum().item()
        if masked > 0:
            action_values[mask] = torch.randint(
                0,
                self.action_space.n,
                (masked,),
                device=state.device,
                dtype=torch.long,
            )
        return action_values

    def load(self, checkpoint_dir):
        pass

    def eval(self):
        self.is_eval = True
        self.qnetwork_local.eval()

    def train(self):
        self.is_eval = False
        self.qnetwork_local.train()
