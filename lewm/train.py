"""Train LeWM.   python -m lewm.train [--epochs 20] [--lambd 0.09]

The loss (official `lejepa_forward`, two terms, one hyperparameter):

    emb, act_emb = encode(all 4 frames, frame-aligned actions)
    pred      = predict(emb[:, :3], act_emb[:, :3])     # teacher-forced
    pred_loss = MSE(pred, emb[:, 1:])                   # targets NOT detached
    loss      = pred_loss + lambda * SIGReg(emb per time step)

No stop-gradient, no EMA, no frozen encoder: gradients flow into the encoder
through predictions AND targets. SIGReg is what makes that survivable —
run `--lambd 0` to watch it fail.

Per-epoch collapse dashboard:
    pred      teacher-forced next-embedding MSE (the number planning tracks)
    sigreg    Gaussianity statistic, unweighted (healthy: settles ~3;
              collapsed: ~50)
    lat_std   mean per-dim latent std — complete collapse drives this to ~0
              while pred "improves" to ~0: the model wins by making the
              problem trivial
    eff_rank  participation ratio of the latent covariance spectrum. Only
              meaningful while lat_std is healthy — under complete collapse
              it measures the rank of residual noise (and can look HIGH).
              Its job is catching dimensional collapse: lat_std fine,
              eff_rank a handful.
"""

from __future__ import annotations

import argparse
import pathlib

import torch
from torch.utils.data import DataLoader

from .data import TrajectorySlices
from .model import LeWM
from .sigreg import SIGReg

CKPT = pathlib.Path("data/ckpt")
MEDIA = pathlib.Path("media")


def compute_loss(model: LeWM, sigreg: SIGReg, obs: torch.Tensor,
                 act: torch.Tensor, lambd: float):
    """obs (B, T, 3, H, W) with T = history+1; act (B, T-1, A)."""
    # Frame-aligned actions: act[:, t] is the action taken AT frame t. The
    # final frame gets a zero placeholder — it is never a prediction input.
    act_padded = torch.cat([act, torch.zeros_like(act[:, :1])], dim=1)
    emb, act_emb = model.encode(obs, act_padded)
    h = model.history
    pred = model.predict(emb[:, :h], act_emb[:, :h])
    pred_loss = (pred - emb[:, 1:]).pow(2).mean()     # no detach: end-to-end
    sig_loss = sigreg(emb.transpose(0, 1))            # per-step over batch
    return pred_loss + lambd * sig_loss, pred_loss, sig_loss


@torch.no_grad()
def latent_stats(emb: torch.Tensor) -> tuple[float, float]:
    z = emb.reshape(-1, emb.size(-1)).float()
    std = z.std(dim=0).mean().item()
    ev = torch.linalg.eigvalsh(torch.cov(z.T))
    eff_rank = (ev.sum() ** 2 / (ev.square().sum() + 1e-12)).item()
    return std, eff_rank


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lambd", type=float, default=0.09)  # the ONE hyperparameter
    ap.add_argument("--lr", type=float, default=1e-4)     # paper: 5e-5 @ 100 ep
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--history", type=int, default=3)
    ap.add_argument("--data", type=str, default="data/reacher")
    ap.add_argument("--tag", type=str, default="reacher")
    args = ap.parse_args()

    torch.manual_seed(3072)  # the official seed, for tradition
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    ds = TrajectorySlices(args.data, k=args.history)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, drop_last=True,
                    num_workers=4, pin_memory=True, persistent_workers=True)

    sample_act = ds[0][1]
    model = LeWM(history=args.history, action_dim=sample_act.shape[-1]).to(dev)
    sigreg = SIGReg().to(dev)
    print(f"windows: {len(ds)}  params: "
          f"{sum(p.numel() for p in model.parameters())/1e6:.2f}M  "
          f"device: {dev}  lambda: {args.lambd}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
    total = args.epochs * len(dl)
    warm = max(1, total // 100)  # 1% linear warmup then cosine, as official
    sched = torch.optim.lr_scheduler.SequentialLR(
        opt,
        [torch.optim.lr_scheduler.LinearLR(opt, 0.01, 1.0, warm),
         torch.optim.lr_scheduler.CosineAnnealingLR(opt, total - warm)],
        milestones=[warm],
    )

    CKPT.mkdir(parents=True, exist_ok=True)
    history = []
    for ep in range(args.epochs):
        sums, n, last = [0.0, 0.0], 0, None
        for obs, act in dl:
            obs = obs.to(dev, non_blocking=True)
            act = act.to(dev, non_blocking=True)
            with torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"):
                loss, pred_l, sig_l = compute_loss(model, sigreg, obs, act, args.lambd)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            sums[0] += pred_l.item() * len(obs)
            sums[1] += sig_l.item() * len(obs)
            n += len(obs)
            last = (obs, act)
        with torch.no_grad():
            emb, _ = model.encode(last[0], torch.cat(
                [last[1], torch.zeros_like(last[1][:, :1])], dim=1))
        std, rank = latent_stats(emb)
        history.append((sums[0] / n, sums[1] / n, std, rank))
        print(f"epoch {ep+1:3d}/{args.epochs}  pred {sums[0]/n:.5f}  "
              f"sigreg {sums[1]/n:.3f}  lat_std {std:.3f}  eff_rank {rank:.1f}")

    path = CKPT / f"{args.tag}.pt"
    torch.save({"model": model.state_dict(), "history": history,
                "lambd": args.lambd, "action_dim": sample_act.shape[-1],
                "history_len": args.history}, path)
    print(f"saved {path}")
    _plot(history, args.tag, args.lambd)


def _plot(history, tag: str, lambd: float) -> None:
    """Dark-surface styling matching the demo videos (validated palette)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    MEDIA.mkdir(exist_ok=True)
    names = ["prediction MSE", "SIGReg statistic", "latent std", "effective rank"]
    fig, axes = plt.subplots(1, 4, figsize=(14, 3), facecolor="#1a1a19")
    for i, (ax, name) in enumerate(zip(axes, names)):
        ax.plot([h[i] for h in history], color="#3987e5", lw=2)
        ax.set_title(name, fontsize=10, color="white")
        ax.set_xlabel("epoch", fontsize=8, color="#c3c2b7")
        ax.set_facecolor("#1a1a19")
        ax.tick_params(colors="#c3c2b7", labelsize=8)
        for s in ax.spines.values():
            s.set_color("#3a3a38")
        ax.grid(alpha=0.15, color="#c3c2b7")
    fig.suptitle(f"LeWM training — {tag} (λ={lambd})", fontsize=11, color="white")
    fig.tight_layout()
    out = MEDIA / f"lewm_{tag}_curves.png"
    fig.savefig(out, dpi=140, facecolor="#1a1a19")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
