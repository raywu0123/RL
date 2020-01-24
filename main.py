import copy

import gym
import numpy as np
from dotenv import load_dotenv
load_dotenv('.env')
import wandb
import json
import torch

from parser import get_parser
from agents import agent_hub
from env_wrappers import EnvWrapperHub


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()

    env = gym.make(args.env)
    env.seed(0)
    print('Original state shape: ', env.observation_space.shape)
    print('Original number of actions: ', env.action_space.n)
    env_wrapper = EnvWrapperHub.get_wrapper(args.env_wrapper_id, is_train=True)
    env = env_wrapper(env)
    print('Wrapped state shape: ', env.observation_space.shape)
    print('Wrapped number of actions: ', env.action_space.n)

    eval_env = copy.deepcopy(env)
    eval_env_wrapper = EnvWrapperHub.get_wrapper(args.env_wrapper_id, is_train=False)
    eval_env = eval_env_wrapper(eval_env)

    agent = agent_hub[args.agent](
        state_size=env.observation_space.shape,
        network_id=args.network_id,
        action_space=env.action_space,
        batch_size=args.batch_size,
        lr=args.lr,
    )
    if args.wandb:
        wandb.init(config=args, project='RL')

    episode_rewards = []
    for i_episode in range(args.n_episode):
        state = torch.from_numpy(env.reset()).float()
        t = 0
        done = False
        rewards = []
        while not done:
            if args.render and i_episode % args.log_episode == 0:
                env.render()

            action = agent.get_action(state)
            next_state, reward, done, info = env.step(action)
            next_state = torch.from_numpy(next_state).float()
            rewards.append(reward)
            agent.end_timestep(
                info,
                state=state,
                action=action,
                reward=reward,
                next_state=next_state,
                done=done,
            )
            state = next_state
            t += 1

        episode_reward = np.sum(rewards)
        episode_rewards.append(episode_reward)
        logs = agent.end_episode()
        all_logs = {
            'episode_reward': episode_reward,
            'episode_length': t,
            **logs,
        }
        if args.wandb:
            wandb.log(all_logs, step=i_episode)
        if i_episode % args.log_freq == 0:
            print(f'episode: {i_episode}, {json.dumps(all_logs)}')

    env.close()
    eval_env.close()
