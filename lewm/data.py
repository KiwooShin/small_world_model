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
        qpos = [env.get_state()[0]]
        for _ in range(steps):
            a = env.scripted_action()
            obs.append(env.step(a))
            acts.append(a)
            qpos.append(env.get_state()[0])
        np.savez_compressed(
            root / f"ep_{ep:05d}.npz",
            obs=(np.stack(obs) * 255).astype(np.uint8),
            action=np.stack(acts).astype(np.float32),
            # Sim poses per frame — never used for training (pixels only);
            # they exist so demo videos can visualize imagined latents by
            # nearest-neighbor retrieval and crisp re-rendering.
            qpos=np.stack(qpos).astype(np.float32),
        )
        if (ep + 1) % 50 == 0:
            print(f"  {ep + 1}/{episodes} episodes")
    return root


class TrajectorySlices(Dataset):
    """Training windows: obs [K+1, 3, H, W] float in [0,1], action [K, fs*A].

    With frameskip fs > 1 (the official LeWM setting is 5), frames are
    strided by fs and the fs raw actions between kept frames concatenate
    into one action BLOCK. This is not a data-efficiency trick — it is what
    keeps goals latent-local: SIGReg whitens the latent space globally, so
    latent distance saturates beyond a short radius (measured: corr with
    true distance falls from 0.35 within 5 cm to ~0.05 past 10 cm), and a
    goal must sit within a few MODEL steps to provide any cost gradient.
    fs=5 turns a 40-env-step goal into an 8-block plan."""

    def __init__(self, root: str | pathlib.Path, k: int, frameskip: int = 1):
        self.k = k
        self.fs = frameskip
        self.files = sorted(pathlib.Path(root).glob("ep_*.npz"))
        if not self.files:
            raise FileNotFoundError(
                f"no episodes under {root}; run `python -m lewm.collect` first")
        self.episodes = []
        self.index: list[tuple[int, int]] = []
        span = k * frameskip
        skipped = 0
        for f in self.files:
            try:
                with np.load(f) as z:
                    ep = (z["obs"], z["action"])
            except Exception:
                skipped += 1        # partially written / corrupt episode
                continue
            fi = len(self.episodes)
            self.episodes.append(ep)
            t = ep[1].shape[0]
            self.index += [(fi, s) for s in range(t - span + 1)]
        if skipped:
            print(f"[data] skipped {skipped} unreadable episode files")

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int):
        fi, s = self.index[i]
        obs, act = self.episodes[fi]
        fs, k = self.fs, self.k
        o = obs[s : s + k * fs + 1 : fs].astype(np.float32) / 255.0
        o = torch.from_numpy(o).permute(0, 3, 1, 2)          # (k+1, 3, H, W)
        a = act[s : s + k * fs].reshape(k, -1)               # (k, fs*A)
        return o, torch.from_numpy(a.copy())
