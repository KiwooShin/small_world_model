"""Offline trajectory data: collection and training windows.

Episode format — the contract shared by every env in this project:
    one .npz file per episode with
        obs:    uint8   [T+1, H, W, 3]   frames, 0..255
        action: float32 [T, A]           action taken between obs[t] and obs[t+1]
The MuJoCo assembly scenes of later milestones emit the same format.
"""

from __future__ import annotations

import pathlib

import numpy as np
import torch
from torch.utils.data import Dataset


def collect(env, root: str | pathlib.Path, episodes: int, steps: int) -> pathlib.Path:
    """Roll `env`'s scripted policy and save episodes under `root`.
    LeWM needs no expert data — "exploratory or pseudo-expert, as long as
    [it] sufficiently cover[s] the environment dynamics" (paper App. E)."""
    root = pathlib.Path(root)
    root.mkdir(parents=True, exist_ok=True)
    for ep in range(episodes):
        obs = [env.reset()]
        acts = []
        for _ in range(steps):
            a = env.scripted_action()
            obs.append(env.step(a))
            acts.append(a)
        np.savez_compressed(
            root / f"ep_{ep:05d}.npz",
            obs=(np.stack(obs) * 255).astype(np.uint8),
            action=np.stack(acts).astype(np.float32),
        )
        if (ep + 1) % 50 == 0:
            print(f"  {ep + 1}/{episodes} episodes")
    return root


class TrajectorySlices(Dataset):
    """Training windows: obs [K+1, 3, H, W] float in [0,1], action [K, A].
    K = number of prediction steps (history size); windows are every valid
    start position of every episode."""

    def __init__(self, root: str | pathlib.Path, k: int):
        self.k = k
        self.files = sorted(pathlib.Path(root).glob("ep_*.npz"))
        if not self.files:
            raise FileNotFoundError(
                f"no episodes under {root}; run `python -m lewm.collect` first")
        self.episodes = []
        self.index: list[tuple[int, int]] = []
        for fi, f in enumerate(self.files):
            with np.load(f) as z:
                self.episodes.append((z["obs"], z["action"]))
            t = self.episodes[-1][1].shape[0]
            self.index += [(fi, s) for s in range(t - k + 1)]

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int):
        fi, s = self.index[i]
        obs, act = self.episodes[fi]
        o = obs[s : s + self.k + 1].astype(np.float32) / 255.0
        o = torch.from_numpy(o).permute(0, 3, 1, 2)
        a = torch.from_numpy(act[s : s + self.k].copy())
        return o, a
