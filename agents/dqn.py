import numpy as np
import random
import copy

import torch
import torch.nn.functional as F
import torch.optim as optim

from .utils import ReplayBuffer
from .base import BaseAgent
from .epsilon_schedulers import scheduler_hub
from networks import network_hub


class DQNAgent(BaseAgent):

    def __init__(
        self,
        state_size,
        network_id: str,
        buffer_size: int = int(20000),
        batch_size: int = 64,
        gamma: float = 0.99,
        lr: float = 1e-4,
        train_freq: int = 4,
        target_update_freq: int = 1000,
        min_epsilon: float = 0.05,
        epsilon_decay: float = 0.0005,
        scheduler_id: str = 'linear',
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.gamma = gamma
        self.train_freq = train_freq
        self.target_update_freq = target_update_freq
        self.epsilon_scheduler = scheduler_hub[scheduler_id](
            1, epsilon_decay, min_epsilon
        )

        # Q-Network
        network = network_hub[network_id](state_size, self.action_space)
        self.qnetwork_local = network.to(self.device)
        self.qnetwork_target = copy.deepcopy(self.qnetwork_local)
        self.optimizer = optim.RMSprop(self.qnetwork_local.parameters(), lr=lr)

        # Replay memory
        self.memory = ReplayBuffer(buffer_size, batch_size, self.device)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0

    def end_timestep(self, info, **kwargs):
        # Save experience in replay memory
        self.t_step += 1
        self.memory.add(**kwargs)

        if len(self.memory) > self.batch_size and self.t_step % self.train_freq == 0:
            experiences = self.memory.sample()
            states, actions, rewards, next_states, dones = experiences
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

            if self.t_step % self.target_update_freq == 0:
                # ------------------- update target network ------------------- #
                self.soft_update(self.qnetwork_local, self.qnetwork_target)

    @staticmethod
    def soft_update(local_model, target_model):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(local_param.data)

    def end_episode(self):
        self.epsilon_scheduler.step()
        return {
            'epsilon': self.epsilon_scheduler.get_epsilon(),
        }

    def get_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > self.epsilon_scheduler.get_epsilon():
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return self.get_random_action()
