"""Data collection for lab 00: rollouts of the scripted policy in PushWorld.

Stored fully in memory as float32 — 200 episodes x 100 steps of 64x64x3 is
~1 GB, trivial on a 121 GB unified-memory machine. Later labs that need more
data stream from disk instead.
"""

from __future__ import annotations

import numpy as np
import torch

from .env import PushWorld


def collect(n_episodes: int, ep_len: int, size: int, seed: int = 0):
    """Returns frames (N, T+1, 3, H, W) float32 and actions (N, T, 2)."""
    env = PushWorld(size=size, seed=seed)
    frames = np.empty((n_episodes, ep_len + 1, 3, size, size), dtype=np.float32)
    actions = np.empty((n_episodes, ep_len, 2), dtype=np.float32)
    for e in range(n_episodes):
        obs = env.reset()
        frames[e, 0] = obs.transpose(2, 0, 1)
        for t in range(ep_len):
            a = env.scripted_action()
            obs = env.step(a)
            actions[e, t] = a
            frames[e, t + 1] = obs.transpose(2, 0, 1)
    return torch.from_numpy(frames), torch.from_numpy(actions)


class TransitionDataset(torch.utils.data.Dataset):
    """(K context frames, action, next frame) tuples over all episodes."""

    def __init__(self, frames: torch.Tensor, actions: torch.Tensor, context: int):
        self.frames, self.actions, self.k = frames, actions, context
        n, t1 = frames.shape[0], frames.shape[1]
        self.per_ep = t1 - context  # valid prediction targets per episode
        self.n = n * self.per_ep

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int):
        e, t = divmod(i, self.per_ep)
        ctx = self.frames[e, t : t + self.k]          # (K, 3, H, W)
        act = self.actions[e, t + self.k - 1]         # action taken after ctx
        nxt = self.frames[e, t + self.k]              # (3, H, W)
        return ctx, act, nxt
