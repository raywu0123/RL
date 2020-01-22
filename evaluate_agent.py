import gym
import json
import torch

from parser import get_parser
from agents import agent_hub
from env_wrappers import EnvWrapperHub
from evaluation import evaluate_agent


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()

    env = gym.make(args.env)
    env.seed(0)

    print('Original state shape: ', env.observation_space.shape)
    print('Original number of actions: ', env.action_space.n)

    env_wrapper = EnvWrapperHub.get_wrapper(args.env_wrapper_id, is_train=False)
    env = env_wrapper(env)

    print('Wrapped state shape: ', env.observation_space.shape)
    print('Wrapped number of actions: ', env.action_space.n)

    agent = agent_hub[args.agent](
        state_size=env.observation_space.shape,
        network_id=args.network_id,
        action_space=env.action_space,
        batch_size=args.batch_size,
        lr=args.lr,
    )
    agent.log(args.checkpoint_dir)
    with torch.no_grad():
        logs = evaluate_agent(agent, env)
        print(json.dumps(logs))
