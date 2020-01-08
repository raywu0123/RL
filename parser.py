from argparse import ArgumentParser


def get_parser():
    parser = ArgumentParser()
    parser.add_argument('-e', '--env', type=str, default='CartPole-v0', help='gym environment_id')
    parser.add_argument('-a', '--agent', type=str, help='agent_id')
    parser.add_argument('-ep', '--n_episode', type=int, default=1000, help='number of episodes')
    parser.add_argument('-le', '--log_episode', type=int, default=10)
    parser.add_argument('-r', '--render', action='store_true')
    return parser
