import gym
import numpy as np

from parser import get_parser
from agents import agent_hub


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()

    env = gym.make(args.env)
    env.seed(0)

    print('State shape: ', env.observation_space.shape)
    print('Number of actions: ', env.action_space.n)

    agent = agent_hub[args.agent](
        state_size=env.observation_space.shape,
        action_space=env.action_space,
        seed=0,
    )

    episode_rewards = []
    for i_episode in range(args.n_episode):
        state = env.reset()
        t = 0
        done = False
        rewards = []
        while not done:
            if args.render and i_episode % args.log_episode == 0:
                env.render()
            action = agent.get_action(state)
            next_state, reward, done, info = env.step(action)
            rewards.append(reward)
            agent.end_timestep(state, next_state, action, done, info, reward)
            state = next_state
            t += 1

        episode_reward = np.sum(rewards)
        episode_rewards.append(episode_reward)
        if i_episode % args.log_episode == 0:
            print(f'episode: {i_episode}, reward: {episode_reward}, length: {t}')

        agent.end_episode()
    env.close()

