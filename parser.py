from argparse import ArgumentParser


def get_parser():
    parser = ArgumentParser()
    parser.add_argument('-e', '--env', type=str, default='CartPole-v0', help='gym environment_id')
    parser.add_argument('-a', '--agent', type=str, help='agent_id')
    parser.add_argument('-ep', '--n_episode', type=int, default=int(1e6), help='number of episodes')
    parser.add_argument('-lf', '--log_freq', type=int, default=100)
    parser.add_argument('-ef', '--evaluate_freq', type=int, default=1000)
    parser.add_argument('-ee', '--evaluate_episodes', type=int, default=100)
    parser.add_argument('-r', '--render', action='store_true')
    parser.add_argument('-bs', '--batch_size', type=int, default=32)
    parser.add_argument('-lr', type=float, default=1e-4)
    parser.add_argument('-nid', '--network_id', type=str)
    parser.add_argument('-ewid', '--env_wrapper_id', type=str, default='identity')
    parser.add_argument('--wandb', action='store_true')

    parser.add_argument('--corpus_name', type=str, default='news')
    parser.add_argument('--vocab_size', type=int, default=5000)
    parser.add_argument('--maxlen', type=int, default=50)
    return parser
