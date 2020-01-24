from argparse import ArgumentParser
import time


def get_parser():
    parser = ArgumentParser()
    parser.add_argument('-e', '--env', type=str, default='CartPole-v0', help='gym environment_id')
    parser.add_argument('-a', '--agent', type=str, help='agent_id')
    parser.add_argument('-ep', '--n_episode', type=int, default=int(1e6), help='number of episodes')
    parser.add_argument('-lf', '--log_freq', type=int, default=100)
    parser.add_argument('--target-update-freq', type=int, default=32000, metavar='τ', help='Number of frames after which to update target network (default: 32,000)')
    parser.add_argument('-ef', '--evaluate_freq', type=int, default=50000)
    parser.add_argument('-ee', '--evaluate_episodes', type=int, default=100)
    parser.add_argument('-r', '--render', action='store_true')
    parser.add_argument('-bs', '--batch_size', type=int, default=32)
    parser.add_argument('-nid', '--network_id', type=str)
    parser.add_argument('-esid', '--epsilon_scheduler_id', type=str, default='linear')
    parser.add_argument('-ed', '--epsilon_decay', type=float, default=1e-6)
    parser.add_argument('-me', '--min_epsilon', type=float, default=0.1)
    parser.add_argument('-ewid', '--env_wrapper_id', type=str, default='identity')
    parser.add_argument('--wandb', action='store_true')

    parser.add_argument('--lr', type=float, default=0.00065, help='learning rate (default: 0.00065)')
    parser.add_argument('--alpha', type=float, default=0.99, help='RMSprop optimizer alpha (default: 0.99)')
    parser.add_argument('--eps', type=float, default=1e-5, help='RMSprop optimizer epsilon (default: 1e-5)')

    parser.add_argument('--gamma', type=float, default=0.99, help='discount factor for rewards (default: 0.99)')
    parser.add_argument('--clip-rewards', action='store_true', default=False, help='Clip rewards to {-1, 0, +1}')
    parser.add_argument('--ale-start-steps', type=int, default=400, help='Number of steps used to initialize ALEs (default: 400)')
    parser.add_argument('--episodic-life', action='store_true', default=False, help='use end of life as end of episode')
    parser.add_argument('--gpu', type=int, default=None, help='GPU ID (default: None)')
    parser.add_argument('--loss-scale', type=str, default=None)
    parser.add_argument('--max-episode-length', type=int, default=18000, help='maximum length of an episode (default: 18,000)')
    parser.add_argument('--max-grad-norm', type=float, default=0.5, help='max norm of gradients (default: 0.5)')
    parser.add_argument('--multiprocessing-distributed', action='store_true',
                        help='Use multi-processing distributed training to launch '
                             'N processes per node, which has N GPUs. This is the '
                             'fastest way to use PyTorch for either single node or '
                             'multi node data parallel training')
    parser.add_argument('--num-ales', type=int, default=16, help='number of environments (default: 16)')
    parser.add_argument('--num-gpus-per-node', type=int, default=-1, help='Number of GPUs per node (default: -1 [use all available])')
    parser.add_argument('--opt-level', type=str, default='O0')
    parser.add_argument('--seed', type=int, default=int(time.time()), help='random seed (default: time())')
    parser.add_argument('--t-max', type=int, default=int(50e6), help='Number of training steps (default: 50,000,000)')
    parser.add_argument('--verbose', action='store_true', default=False, help='Enable verbose logging')
    parser.add_argument('--use-adam', action='store_true', default=False, help='use ADAM optimizer')
    parser.add_argument('--use-cuda-env', action='store_true', default=False, help='use CUDA for ALE updates')
    parser.add_argument('--use-openai', action='store_true', default=False, help='Use OpenAI Gym environment')
    parser.add_argument('--use-openai-test-env', action='store_true', default=False, help='Use OpenAI Gym test environment')
    parser.add_argument('--history-length', type=int, default=4, help='Number of consecutive states processed')
    parser.add_argument('--model', type=str, metavar='PARAMS', help='Pretrained model (state dict)')
    parser.add_argument('--memory-capacity', type=int, default=int(10e5), metavar='CAPACITY', help='Experience replay memory capacity (default: 1,000,000)')
    parser.add_argument('--replay-frequency', type=int, default=4, metavar='k', help='The number of the gradient step updates (intensity) per step in the environment')
    parser.add_argument('--priority-exponent', type=float, default=0.7, metavar='ω', help='Prioritised experience replay exponent (originally denoted α)')
    parser.add_argument('--priority-weight', type=float, default=0.5, metavar='β', help='Initial prioritised experience replay importance sampling weight')
    parser.add_argument('--multi-step', type=int, default=3, metavar='n', help='Number of steps for multi-step return (default: 3)')
    parser.add_argument('--reward-clip', action='store_true', default=False, help='Clip rewards to {-1, 0, +1}')
    parser.add_argument('--adam-eps', type=float, default=1.5e-4, metavar='ε', help='Adam epsilon')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size (default: 32)')
    parser.add_argument('--learn-start', type=int, default=int(40e3), help='Number of steps before starting training (default: 40,000)')
    parser.add_argument('--evaluation-size', type=int, default=500, help='Number of transitions to use for validating Q')
    parser.add_argument('--priority-replay', action='store_true', default=False, help='Enable prioritized experience replay')
    return parser
