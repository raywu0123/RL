from datetime import datetime
import math

import numpy as np
import pytz
import torch
from tqdm import tqdm
from torchcule.atari import Env as AtariEnv
from dotenv import load_dotenv
load_dotenv('.env')
import wandb

from agents import agent_hub
from epsilon_schedulers import scheduler_hub
from utils import DataPrefetcher, ReplayMemory
from evaluation import test


def format_time(f):
    return datetime.fromtimestamp(f, tz=pytz.utc).strftime('%H:%M:%S.%f s')


def get_env(args, train_device):
    env = AtariEnv(
        args.env,
        args.num_ales,
        color_mode='gray',
        device=train_device,
        rescale=True,
        clip_rewards=args.reward_clip,
        episodic_life=True,
        repeat_prob=0.0,
    )
    env.train()
    observation = env.reset(
        initial_steps=args.ale_start_steps,
        verbose=args.verbose,
    ).clone().squeeze(-1)
    return env, observation


def worker(args):
    args.use_cuda_env = args.use_cuda_env and torch.cuda.is_available()
    args.verbose = args.verbose

    train_device = torch.device('cuda', 0) if torch.cuda.is_available() else torch.device('cpu')
    print(f'Training on device: {train_device}')

    print('Setting Random Seeds...')
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.use_cuda_env:
        torch.cuda.manual_seed(args.seed)

    print('Initializing environments...')
    env, observation = get_env(args, train_device)
    print('Original state shape: ', env.observation_space.shape)
    print('Original number of actions: ', env.action_space.n)

    # Agent
    dqn = agent_hub[args.agent](
        state_size=env.observation_space.shape,
        history_length=args.history_length,
        network_id=args.network_id,
        action_space=env.action_space,
        batch_size=args.batch_size,
        device=train_device,
        lr=args.lr,
    )
    mem = ReplayMemory(args, args.memory_capacity, train_device)
    mem.reset(observation)
    prefetcher = DataPrefetcher(args.batch_size, train_device, mem)
    priority_weight_increase = (1 - args.priority_weight) / (args.t_max - args.learn_start)

    state = torch.zeros(
        (args.num_ales, args.history_length, 84, 84),
        device=train_device,
        dtype=torch.float32,
    )
    state[:, -1] = observation.to(device=train_device, dtype=torch.float32).div_(255.0)

    epsilon_scheduler = scheduler_hub[args.epsilon_scheduler_id](
        1, args.epsilon_decay, args.min_epsilon
    )
    num_frames_per_iter = args.num_ales
    total_steps = math.ceil(args.t_max / num_frames_per_iter)

    print("Initializing Validation Resources")
    test_env = AtariEnv(
        args.env,
        args.evaluate_episodes,
        color_mode='gray',
        device='cpu',
        rescale=True,
        clip_rewards=False,
        episodic_life=False,
        repeat_prob=0.0,
        frameskip=4,
    )

    print('Initializing offsets')
    eval_offset = 0
    target_update_offset = 0

    if args.wandb:
        wandb.init(config=args, project='RL')

    print('Entering main training loop')
    iterator = tqdm(range(total_steps))
    env_stream = torch.cuda.Stream()
    for update in iterator:
        T = update * num_frames_per_iter
        if T >= args.learn_start:
            epsilon_scheduler.step(num=num_frames_per_iter)

        epsilon = epsilon_scheduler.get_epsilon()
        dqn.eval()
        # Choose an action greedily (with noisy weights)
        action = dqn.act(state, epsilon=epsilon)
        dqn.train()

        torch.cuda.synchronize(device=train_device)
        with torch.cuda.stream(env_stream):
            observation, reward, done, info = env.step(action)  # Step
            observation = observation.clone().squeeze(-1)
            observation = observation.float().div_(255.0)
            state[:, :-1].copy_(state[:, 1:].clone())
            not_done = 1.0 - done.float()
            state *= not_done.view(-1, 1, 1, 1)
            state[:, -1].copy_(observation)

            avg_loss = 0.0
        if T >= args.learn_start:
            # Anneal importance sampling weight β to 1
            mem.priority_weight = min(mem.priority_weight + priority_weight_increase, 1)
            prefetcher.preload()

            num_minibatches = min(int(args.num_ales / args.replay_frequency), 8)
            for _ in range(num_minibatches):
                # Sample transitions
                idxs, states, actions, returns, next_states, nonterminals, weights = \
                    prefetcher.next()
                loss = dqn.learn(states, actions, returns, next_states, nonterminals)
                mem.update_priorities(idxs, loss)  # Update priorities of sampled transitions

                avg_loss += loss.mean().item()
            avg_loss /= num_minibatches

        if T >= target_update_offset:
            dqn.update_target_net()
            target_update_offset += args.target_update_freq

        if args.wandb:
            wandb.log({
                'epsilon': epsilon,
                'loss': avg_loss,
            }, step=T)

        if T >= eval_offset:
            dqn.eval()
            with torch.no_grad():
                eval_rewards, eval_lengths = test(args, dqn, test_env, train_device)
            eval_mean_reward = eval_rewards.mean().item()
            eval_mean_length = eval_lengths.mean().item()
            dqn.train()
            eval_offset += args.evaluate_freq
            if args.wandb:
                wandb.log({
                    'eval_mean_reward': eval_mean_reward,
                    'episode_length': eval_mean_length,
                }, step=T)

        torch.cuda.current_stream().wait_stream(env_stream)
        mem.append(observation, action, reward, done)  # Append transition to memory
        progress_data = f'T = {T} ' \
                        f'epsilon = {epsilon:.4f} ' \
                        f'loss: {avg_loss:.4f} ' \
                        f'mean-reward: {eval_mean_reward:2.2f}'
        iterator.set_postfix_str(progress_data)


if __name__ == '__main__':
    from parser import get_parser
    p = get_parser()
    args = p.parse_args()
    worker(args)
