from argparse import ArgumentParser


def get_parser():
    parser = ArgumentParser()
    parser.add_argument('-e', '--env', type=str, default='CartPole-v0', help='gym environment_id')
    parser.add_argument('-a', '--agent', type=str, help='agent_id')
    parser.add_argument('-ep', '--n_episode', type=int, default=int(1e6), help='number of episodes')
    parser.add_argument('-le', '--log_episode', type=int, default=10)
    parser.add_argument('-r', '--render', action='store_true')
    parser.add_argument('-bs', '--batch_size', type=int, default=64)
    parser.add_argument('-lr', type=float, default=1e-4)
    parser.add_argument('-nid', '--network_id', type=str)
    parser.add_argument('-ipid', '--input_pipe_id', type=str, default='identity')
    parser.add_argument('--wandb', action='store_true')
    return parser
