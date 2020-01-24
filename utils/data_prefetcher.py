import torch


class DataPrefetcher:

    def __init__(self, batch_size, device, mem):
        self.batch_size = batch_size
        self.device = device
        self.mem = mem
        self.stream = torch.cuda.Stream()

    def preload(self):
        with torch.cuda.stream(self.stream):
            (
                self.idxs,
                self.states,
                self.actions,
                self.returns,
                self.next_states,
                self.nonterminals,
                self.weights,
            ) = self.mem.sample(self.batch_size)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)

        idxs = self.idxs.to(device=self.device)
        states = self.states.to(device=self.device)
        actions = self.actions.to(device=self.device)
        returns = self.returns.to(device=self.device)
        next_states = self.next_states.to(device=self.device)
        nonterminals = self.nonterminals.to(device=self.device)
        weights = self.weights.to(device=self.device)

        self.preload()
        return idxs, states, actions, returns, next_states, nonterminals, weights
