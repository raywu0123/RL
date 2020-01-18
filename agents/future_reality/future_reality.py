import numpy as np
import random
import copy

import torch
from torch import nn
import torch.nn.functional as F
from torch import optim

from agents.utils import ReplayBuffer
from agents.base import BaseAgent
from agents.epsilon_schedulers import scheduler_hub
from .future_network import FutureNetwork
from .reward_network import RewardNetwork
from .state_ae import StateEncoder, StateDecoder
from .keypoint_network import KeypointNetwork


class FutureRealityAgent(BaseAgent):

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
        self.lr = lr
        self.train_freq = train_freq
        self.epsilon_scheduler = scheduler_hub[scheduler_id](
            1, epsilon_decay, min_epsilon
        )
        self.t_step = 0
        self.memory = ReplayBuffer(buffer_size, batch_size, self.device)
        self.episode_logs = {}

        # self.future_network = FutureNetwork()
        # self.reward_network = RewardNetwork()
        # self.opt_future_reward = optim.RMSprop(self.future_network.parameters(), lr=self.lr)
        # self.keypoint_network = KeypointNetwork()
        # self.opt_keypoint = optim.RMSprop(self.future_network.parameters(), lr=self.lr)
        self.state_encoder = StateEncoder(state_size)
        self.state_decoder = StateDecoder(state_size)
        self.opt_ae = optim.RMSprop(
            list(self.state_encoder.parameters()) + list(self.state_decoder.parameters()),
            lr=self.lr,
        )

    def end_timestep(self, info, **kwargs):
        # Save experience in replay memory
        self.t_step += 1
        self.memory.add(**kwargs)

        if len(self.memory) > self.batch_size and self.t_step % self.train_freq == 0:
            experiences = self.memory.sample()
            states, actions, rewards, next_states, dones = experiences

            # train state autoencoder
            self.opt_ae.zero_grad()
            ae_loss, latent_states = self.state_autoencoder_loss(states)
            ae_loss.backward()
            self.opt_ae.step()

            # self.state_encoder.eval()
            # # train future & reward network
            # self.opt_future_reward.zero_grad()
            # future_loss = self.future_network_loss(latent_states)
            # reward_loss = self.reward_network_loss(latent_states, rewards)
            # total_loss = future_loss + reward_loss
            # total_loss.backward()
            # self.opt_future_reward.step()
            # self.state_encoder.train()

            self.episode_logs = {
                **self.episode_logs,
                'ae_loss': ae_loss.item(),
            }

    def state_autoencoder_loss(self, states):
        latent_states = self.state_encoder(states)
        recon_states = self.state_decoder(latent_states)
        loss = nn.MSELoss()(recon_states, states)
        return loss, latent_states

    def future_network_loss(self, latent_states):
        reality_fn, _ = self.future_network(latent_states)
        keypoints = self.keypoint_network(latent_states)
        future_states = reality_fn(keypoints)
        accumulated_rewards = self.reward_network(future_states)
        loss = -accumulated_rewards  # maximize
        return loss

    def reward_network_loss(self, latent_states, rewards):
        pred_rewards = self.reward_network(latent_states)
        loss = nn.MSELoss()(rewards, pred_rewards)
        return loss

    def keypoint_network_loss(self, latent_state, rewards):
        keypoint_ground_truths = self.get_keypoints(rewards)
        keypoint_preds = self.keypoint_network(latent_state)
        loss = nn.MSELoss()(keypoint_ground_truths, keypoint_preds)
        return loss

    @staticmethod
    def get_keypoints(rewards):
        pass

    def end_episode(self) -> dict:
        self.epsilon_scheduler.step()
        # train keypoint_network
        # TODO
        episode_logs = copy.deepcopy(self.episode_logs)
        self.episode_logs.clear()
        return {
            **episode_logs,
            'epsilon': self.epsilon_scheduler.get_epsilon(),
        }

    def get_action(self, state):
        # state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        # self.future_network.eval()
        # self.state_encoder.eval()
        # with torch.no_grad():
        #     latent_state = self.state_encoder(state)
        #     _, p_action = self.future_network(latent_state)
        # self.future_network.train()
        # self.state_encoder.train()
        #
        # # Epsilon-greedy action selection
        # if random.random() > self.epsilon_scheduler.get_epsilon():
        #     return np.argmax(p_action.cpu().data.numpy())
        # else:
        return self.get_random_action()


