"""Train the DINO-WM baseline.  python -m lewm.train_dinowm [--env reacher]

Mirrors lewm.train minus SIGReg and the collapse dashboard — a frozen
encoder cannot collapse; only the patch-grid predictor trains. Reference
config uses lr 5e-4, wd 0 (vs LeWM's 5e-5 / 1e-3)."""

from __future__ import annotations

import argparse
import pathlib

import torch
from torch.utils.data import DataLoader

from .data import TrajectorySlices
from .dinowm import DinoWM, compute_loss

CKPT = pathlib.Path("data/ckpt")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--history", type=int, default=3)
    ap.add_argument("--frameskip", type=int, default=5)
    ap.add_argument("--env", type=str, default="reacher")
    ap.add_argument("--data", type=str, default=None)
    ap.add_argument("--tag", type=str, default=None)
    args = ap.parse_args()

    torch.manual_seed(42)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    args.data = args.data or f"data/{args.env}"
    args.tag = args.tag or f"{args.env}_dinowm"
    ds = TrajectorySlices(args.data, k=args.history, frameskip=args.frameskip)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, drop_last=True,
                    num_workers=4, pin_memory=True, persistent_workers=True)

    a_dim = ds[0][1].shape[-1]
    model = DinoWM(action_dim=a_dim, history=args.history).to(dev)
    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"windows: {len(ds)}  trainable params: "
          f"{sum(p.numel() for p in trainable)/1e6:.2f}M  device: {dev}")

    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, args.epochs * len(dl))

    CKPT.mkdir(parents=True, exist_ok=True)
    history = []
    for ep in range(args.epochs):
        tot, n = 0.0, 0
        for obs, act in dl:
            obs = obs.to(dev, non_blocking=True)
            act = act.to(dev, non_blocking=True)
            with torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"):
                loss = compute_loss(model, obs, act)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step()
            sched.step()
            tot += loss.item() * len(obs)
            n += len(obs)
        history.append(tot / n)
        print(f"epoch {ep+1:3d}/{args.epochs}  pred {tot/n:.5f}")

    path = CKPT / f"{args.tag}.pt"
    torch.save({"model": model.state_dict(), "history": history,
                "action_dim": a_dim, "history_len": args.history,
                "env": args.env, "frameskip": args.frameskip,
                "model_type": "dinowm"}, path)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
