"""Offline trajectory data for LeWM training.

This module is COMPLETE — it is infrastructure, not part of the exercise.

Episode format (the contract everything else in this lab builds on):
    one .npz file per episode with
        obs:    uint8   [T+1, H, W, 3]   frames, 0..255
        action: float32 [T, A]           action taken between obs[t] and obs[t+1]

The same format is what your MuJoCo dumper should emit later for the
LeWM-vs-DINO-WM comparison: render each step, stack, save. Nothing here
assumes PushWorld beyond the `collect_pushworld` helper.
"""

from __future__ import annotations

import pathlib

import numpy as np
import torch
from torch.utils.data import Dataset

from labs.lab00_dynamics.env import PushWorld


def collect_pushworld(root: str | pathlib.Path, episodes: int = 400, steps: int = 64,
                      seed: int = 0) -> pathlib.Path:
    """Collect random-policy PushWorld episodes as .npz files under `root`.

    Random actions are fine here: LeWM trains on offline trajectories and the
    paper's datasets are exploratory, not expert. ~400 x 64 steps trains a
    smoke-test model; scale up once your implementation passes the checks.
    """
    root = pathlib.Path(root)
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    env = PushWorld(seed=seed)
    for ep in range(episodes):
        obs = [env.reset()]
        acts = []
        for _ in range(steps):
            a = rng.uniform(-1.0, 1.0, size=2).astype(np.float32)
            obs.append(env.step(a))
            acts.append(a)
        np.savez_compressed(
            root / f"ep_{ep:05d}.npz",
            obs=(np.stack(obs) * 255).astype(np.uint8),
            action=np.stack(acts),
        )
    return root


class TrajectorySlices(Dataset):
    """Yields training windows: obs [K+1, 3, H, W] float in [0,1], action [K, A].

    K is the number of prediction steps the model will be trained on
    (teacher-forced multi-step). Windows are sampled from every valid start
    position of every episode.
    """

    def __init__(self, root: str | pathlib.Path, k: int):
        self.k = k
        self.files = sorted(pathlib.Path(root).glob("ep_*.npz"))
        if not self.files:
            raise FileNotFoundError(f"no episodes under {root}; run collect_pushworld first")
        self.index: list[tuple[int, int]] = []
        self.episodes = []
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
        o = torch.from_numpy(o).permute(0, 3, 1, 2)          # [K+1, 3, H, W]
        a = torch.from_numpy(act[s : s + self.k].copy())      # [K, A]
        return o, a
