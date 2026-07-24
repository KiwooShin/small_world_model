"""Lab 00: train the naive pixel dynamics model and render the demo.

Usage:
    python -m labs.lab00_dynamics.run                 # full run (~5 min on GB10)
    python -m labs.lab00_dynamics.run --smoke         # 1-minute sanity run

Outputs (media/):
    lab00_rollout.gif   ground truth | model imagination | |error|, side by side
    lab00_psnr.png      PSNR vs rollout horizon (the compounding-error curve)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .data import TransitionDataset, collect
from .env import PushWorld
from .model import PixelDynamics

MEDIA = Path(__file__).resolve().parents[2] / "media"


def train(model, loader, device, epochs, lr=3e-4):
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()
    for ep in range(epochs):
        total, count = 0.0, 0
        for ctx, act, nxt in loader:
            ctx, act, nxt = ctx.to(device), act.to(device), nxt.to(device)
            pred = model(ctx, act)
            loss = F.mse_loss(pred, nxt)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += loss.item() * ctx.shape[0]
            count += ctx.shape[0]
        print(f"epoch {ep + 1}/{epochs}  mse {total / count:.6f}")


@torch.no_grad()
def rollout(model, env: PushWorld, context: int, horizon: int, device):
    """Autoregressive rollout: seed with `context` real frames, then feed the
    model its own predictions. Returns (gt, pred) as (T, H, W, 3) arrays and
    the shared action sequence."""
    model.eval()
    obs = env.reset()
    gt = [obs]
    actions = []
    # Seed frames come from the real env for both branches.
    for _ in range(context - 1):
        a = env.scripted_action()
        actions.append(a)
        gt.append(env.step(a))
    # Ground-truth branch continues in the real env.
    for _ in range(horizon):
        a = env.scripted_action()
        actions.append(a)
        gt.append(env.step(a))
    gt = np.stack(gt)  # (context + horizon, H, W, 3)

    # Model branch: same seed frames, same actions, but predictions feed back.
    window = [torch.from_numpy(f.transpose(2, 0, 1)) for f in gt[:context]]
    preds = list(gt[:context])
    for t in range(horizon):
        ctx = torch.stack(window[-context:]).unsqueeze(0).to(device)
        act = torch.from_numpy(
            np.asarray(actions[context - 1 + t], dtype=np.float32)
        ).unsqueeze(0).to(device)
        nxt = model(ctx, act)[0].cpu()
        window.append(nxt)
        preds.append(nxt.numpy().transpose(1, 2, 0))
    return gt, np.stack(preds), context


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a - b) ** 2))
    return 99.0 if mse < 1e-10 else -10.0 * np.log10(mse)


def render_gif(gt, pred, context, path: Path, scale: int = 3, fps: int = 12):
    """Side-by-side (GT | imagination | error) GIF with labels and a red
    border on the imagination panel once autoregression starts."""
    import imageio.v2 as imageio
    from PIL import Image, ImageDraw

    frames = []
    h, w = gt.shape[1:3]
    for t in range(gt.shape[0]):
        err = np.abs(gt[t] - pred[t]).mean(axis=-1)
        err_rgb = np.stack([err * 3, err * 0.6, err * 0.6], axis=-1).clip(0, 1)
        panels = [gt[t], pred[t].clip(0, 1), err_rgb]
        row = np.concatenate(panels, axis=1)
        img = Image.fromarray((row * 255).astype(np.uint8)).resize(
            (3 * w * scale, h * scale), Image.NEAREST
        )
        d = ImageDraw.Draw(img)
        imagining = t >= context
        for i, label in enumerate(("ground truth", "imagination", "|error|")):
            d.text((i * w * scale + 5, 3), label, fill=(255, 255, 255))
        d.text((5, h * scale - 14), f"t={t:03d}" + ("  [autoregressive]" if imagining else "  [seed]"),
               fill=(255, 210, 80) if imagining else (160, 160, 160))
        if imagining:  # red border on the imagination panel
            x0, x1 = w * scale, 2 * w * scale - 1
            d.rectangle([x0, 0, x1, h * scale - 1], outline=(255, 60, 60), width=2)
        frames.append(np.asarray(img))
    imageio.mimsave(path, frames, fps=fps, loop=0)
    print(f"wrote {path}")


def render_psnr(curves: list[np.ndarray], context: int, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    arr = np.stack(curves)  # (episodes, horizon)
    mean, lo, hi = arr.mean(0), arr.min(0), arr.max(0)
    fig, ax = plt.subplots(figsize=(6, 3.5), dpi=140)
    x = np.arange(1, arr.shape[1] + 1)
    ax.plot(x, mean, color="#e07020", lw=2, label="mean PSNR")
    ax.fill_between(x, lo, hi, color="#e07020", alpha=0.2, label="min–max")
    ax.set_xlabel(f"autoregressive steps after {context} seed frames")
    ax.set_ylabel("PSNR (dB)")
    ax.set_title("Lab 00: naive pixel dynamics — error compounds with horizon")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path)
    print(f"wrote {path}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--episodes", type=int, default=200)
    p.add_argument("--ep-len", type=int, default=100)
    p.add_argument("--size", type=int, default=64)
    p.add_argument("--context", type=int, default=2)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--horizon", type=int, default=60, help="rollout length for the demo")
    p.add_argument("--eval-episodes", type=int, default=8)
    p.add_argument("--smoke", action="store_true", help="tiny 1-minute run")
    args = p.parse_args()
    if args.smoke:
        args.episodes, args.epochs, args.horizon, args.eval_episodes = 20, 2, 30, 2

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)

    print(f"collecting {args.episodes} episodes...")
    frames, actions = collect(args.episodes, args.ep_len, args.size, seed=0)
    ds = TransitionDataset(frames, actions, args.context)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=args.batch, shuffle=True, num_workers=2, drop_last=True
    )

    model = PixelDynamics(context=args.context).to(device)
    n_params = sum(x.numel() for x in model.parameters())
    print(f"model params: {n_params / 1e6:.2f}M  device: {device}")
    train(model, loader, device, args.epochs)

    MEDIA.mkdir(exist_ok=True)
    # Demo GIF from one held-out episode (fresh env seed).
    env = PushWorld(size=args.size, seed=1234)
    gt, pred, ctx = rollout(model, env, args.context, args.horizon, device)
    render_gif(gt, pred, ctx, MEDIA / "lab00_rollout.gif")

    # PSNR-vs-horizon over several held-out episodes.
    curves = []
    for e in range(args.eval_episodes):
        env = PushWorld(size=args.size, seed=5000 + e)
        gt, pred, ctx = rollout(model, env, args.context, args.horizon, device)
        curves.append(np.array([psnr(gt[ctx + t], pred[ctx + t]) for t in range(args.horizon)]))
    render_psnr(curves, args.context, MEDIA / "lab00_psnr.png")
    print(f"final-step mean PSNR: {np.stack(curves)[:, -1].mean():.2f} dB "
          f"(first-step: {np.stack(curves)[:, 0].mean():.2f} dB)")


if __name__ == "__main__":
    main()
