from collections import Counter, defaultdict
from itertools import chain
import os
from argparse import Namespace

from gym import Env, spaces
import numpy as np
from tqdm import tqdm

from .utils import crop_or_pad_to_maxlen


class TextPuzzle(Env):

    corpus_path_dict = {
        'news': os.getenv('NEWS_CORPUS_PATH'),
    }

    def __init__(self, args: Namespace):
        self.corpus_name = args.corpus_name
        self.vocab_size = args.vocab_size
        self.maxlen = args.maxlen

        self.corpus_path = self.corpus_path_dict[self.corpus_name]

        self.pad_idx = 0
        self._load_corpus(self.corpus_path, self.vocab_size, self.maxlen)

    @staticmethod
    def _load_corpus(path, vocab_size: int, maxlen: int):
        print('Loading corpus...')
        with open(path, 'r') as f_in:
            corpus = [
                line.rstrip().lower().split()
                for line in tqdm(f_in)
            ]
        counter = Counter(chain(*corpus))
        tokens = counter.most_common(vocab_size - 1)  # reserve for pad

        token2idx = defaultdict(int)
        for idx, (token, _) in enumerate(tokens):
            token2idx[token] = idx + 1

        idx2token = {idx: token for idx, token in enumerate(tokens)}
        idx2token[0] = '[PAD]'

        corpus = np.asarray([
            crop_or_pad_to_maxlen([token2idx[token] for token in sentence], maxlen)
            for sentence in corpus
        ])
        print('Complete')
        return corpus, token2idx, idx2token

    def step(self, action):
        pass

    def reset(self):
        pass

    def render(self, mode='human'):
        pass

    @property
    def observation_space(self):
        return spaces.Space(shape=(self.maxlen,), dtype=np.int32)

    @property
    def action_space(self):
        return spaces.Discrete(n=self.vocab_size)
