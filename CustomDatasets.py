import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


# predict future states
class LookaheadDataset(Dataset):
    def __init__(self, states, actions, n_ahead=1):
        """
        Class to predict future states given initial state and all actions.
        In other words, if we want to predict n states ahead, and states are (s_0, s_1, ..., s_n), with actions (a_0, a_1, ... a_n).
        Given s_0, (a_0, ..., a_n), predict d_1, d_2, ... d_n, where d_i = s_i - s_0
        Arguments:
            states (np.ndarray): array of states over time. Size: (rollout, time step, state)
            actions (np.ndarray): array of actions over time. Size: (rollout, time step, action)
            n_ahead (int): number of states ahead to predict
        """
        super().__init__()
        self.actions = actions
        self.states = states
        self.n_ahead = n_ahead

        # n is number of rollouts, t is time steps
        n, t, _ = self.actions.shape
        self.n = n
        self.t = t
        self.k = self.t - self.n_ahead

    def __getitem__(self, index):
        n = index // self.k
        t = index % self.k

        actions = self.actions[n, t : t + self.n_ahead]
        state_t = self.states[n, t]
        state_t1 = self.states[n, t + 1 : t + 1 + self.n_ahead]
        delta_t = state_t1 - state_t

        return actions, state_t, delta_t

    def __len__(self):
        return self.n * self.k


if __name__ == "__main__":
    from load_data import *

    p, x = load_error_data()
    n_ahead = 2
    lookahead_dataset = LookaheadDataset(states=x, actions=p, n_ahead=n_ahead)
    pk, xk, dk = lookahead_dataset[193829]
    print(pk.shape, xk.shape, dk.shape)
