"""Collect offline episodes.   python -m lewm.collect [--episodes 500]"""

from __future__ import annotations

import argparse

from .data import collect
from .envs.reacher import Reacher


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=500)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--out", type=str, default="data/reacher")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    env = Reacher(seed=args.seed)
    print(f"collecting {args.episodes} episodes x {args.steps} steps -> {args.out}")
    collect(env, args.out, args.episodes, args.steps)
    print("done")


if __name__ == "__main__":
    main()
