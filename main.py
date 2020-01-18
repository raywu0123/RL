import os

import gym
import numpy as np
from dotenv import load_dotenv
load_dotenv('.env')
import wandb
import json

from parser import get_parser
from agents import agent_hub
from input_pipes import input_pipe_hub
from networks import network_hub


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()

    env = gym.make(args.env)
    env.seed(0)

    print('State shape: ', env.observation_space.shape)
    print('Number of actions: ', env.action_space.n)

    input_pipe = input_pipe_hub[args.input_pipe_id](env.observation_space.shape)
    network = network_hub[args.network_id](input_pipe.get_state_size(), env.action_space,)

    agent = agent_hub[args.agent](
        action_space=env.action_space,
        batch_size=args.batch_size,
        lr=args.lr,
        network=network,
    )
    if args.wandb:
        wandb.login(key=os.environ.get('WANDB_API_KEY'))
        wandb.init(config=args)

    episode_rewards = []
    for i_episode in range(args.n_episode):
        state = env.reset()
        state = input_pipe(state)

        t = 0
        done = False
        rewards = []
        while not done:
            if args.render and i_episode % args.log_episode == 0:
                env.render()

            action = agent.get_action(state)
            next_state, reward, done, info = env.step(action)
            next_state = input_pipe(next_state)
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
        if i_episode % args.log_episode == 0:
            print(f'episode: {i_episode}, {json.dumps(all_logs)}')

        input_pipe.reset()
    env.close()
