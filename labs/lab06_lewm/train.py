"""LeWM training.  The loop is provided; the LOSS is yours.     ★ TASK 3 ★

Usage:
    python -m labs.lab06_lewm.train                # train with SIGReg
    python -m labs.lab06_lewm.train --lambd 0.0    # ablation: watch it collapse
    python -m labs.lab06_lewm.train --epochs 5     # quick smoke run

Per-epoch diagnostics explained (this is your collapse dashboard):
    pred      teacher-forced next-embedding MSE
    sigreg    the Gaussianity statistic (before weighting by lambda)
    lat_std   mean per-dimension std of the latents — complete collapse
              drives this to ~0
    eff_rank  participation ratio of the latent covariance spectrum,
              (sum eigvals)^2 / sum(eigvals^2), in [1, 192] — dimensional
              collapse shows up here while lat_std still looks healthy
A healthy run: pred falls, sigreg stays low, lat_std ~ 1, eff_rank high
(tens). A collapsing run (--lambd 0.0): pred falls to ~0 *because* lat_std
does — the model wins by making the problem trivial.
"""

from __future__ import annotations

import argparse
import pathlib

import torch
from torch.utils.data import DataLoader

from .data import TrajectorySlices, collect_pushworld
from .model import LeWM
from .sigreg import SIGReg

DATA = pathlib.Path("data/lab06_pushworld")
CKPT = pathlib.Path("data/lab06_ckpt")


def compute_loss(model: LeWM, sigreg: SIGReg, obs: torch.Tensor,
                 act: torch.Tensor, lambd: float):
    """The complete LeWM objective. Returns (loss, pred_loss, sigreg_loss).

    obs: (B, T, 3, H, W) with T = history + 1 = 4 frames
    act: (B, T-1, A) actions between consecutive frames

    Mirror the official `lejepa_forward` exactly:

    1. Actions: the model wants one action per frame — the action *taken at*
       that frame. act[:, t] is the action between frames t and t+1, so the
       frame-aligned action tensor is just act itself, and only the first
       `history` frames need actions (the last frame is never a prediction
       input). Pad or slice accordingly when you call encode: encode wants
       (B, T, A); passing act padded with one zero row at the end is fine
       (that row is never attended to by a prediction target).

    2. Encode ALL T frames (and the padded actions) in one call ->
       emb (B, 4, 192), act_emb (B, 4, 192).

    3. Prediction loss, teacher-forced:
         context   = first `history` = 3 embeddings and their action embs
         predictions = model.predict(context...)         # (B, 3, 192)
         targets     = emb[:, 1:]                        # (B, 3, 192)
         pred_loss   = MSE
       THE CRITICAL DETAIL: do **not** detach the targets. No stop-gradient,
       no EMA — gradients flow into the encoder through BOTH the prediction
       and the target. That is the paper's headline claim, and it is exactly
       what SIGReg makes survivable. (check.py verifies gradients reach the
       target path.)

    4. SIGReg on the *encoder* embeddings only (never on predictions),
       applied per time step across the batch: the statistic wants shape
       (T, B, D), your emb is (B, T, D) — transpose accordingly.

    5. loss = pred_loss + lambd * sigreg_loss     (lambda = 0.09 default)
    """
    raise NotImplementedError("TASK 3: implement compute_loss")


# ----------------------------------------------------------- provided loop --

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
    ap.add_argument("--tag", type=str, default="lewm")
    args = ap.parse_args()

    torch.manual_seed(3072)  # the official seed, for tradition
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    if not DATA.exists():
        print("collecting PushWorld episodes...")
        collect_pushworld(DATA)
    ds = TrajectorySlices(DATA, k=args.history)  # windows of history+1 frames
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, drop_last=True,
                    num_workers=4, pin_memory=True)

    model = LeWM(history=args.history).to(dev)
    sigreg = SIGReg().to(dev)
    print(f"params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M  "
          f"device: {dev}  lambda: {args.lambd}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
    total = args.epochs * len(dl)
    warm = max(1, total // 100)  # 1% linear warmup, then cosine — as official
    sched = torch.optim.lr_scheduler.SequentialLR(
        opt,
        [torch.optim.lr_scheduler.LinearLR(opt, 0.01, 1.0, warm),
         torch.optim.lr_scheduler.CosineAnnealingLR(opt, total - warm)],
        milestones=[warm],
    )

    CKPT.mkdir(parents=True, exist_ok=True)
    history = []
    for ep in range(args.epochs):
        sums, n = [0.0, 0.0], 0
        last_emb = None
        for obs, act in dl:
            obs, act = obs.to(dev, non_blocking=True), act.to(dev, non_blocking=True)
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
            if n >= len(ds) - args.batch:  # keep last batch's embs for stats
                with torch.no_grad():
                    last_emb, _ = model.encode(obs, torch.cat(
                        [act, torch.zeros_like(act[:, :1])], dim=1))
        std, rank = latent_stats(last_emb)
        history.append((sums[0] / n, sums[1] / n, std, rank))
        print(f"epoch {ep+1:3d}/{args.epochs}  pred {sums[0]/n:.5f}  "
              f"sigreg {sums[1]/n:.3f}  lat_std {std:.3f}  eff_rank {rank:.1f}")

    torch.save({"model": model.state_dict(), "history": history,
                "lambd": args.lambd}, CKPT / f"{args.tag}.pt")
    print(f"saved {CKPT / (args.tag + '.pt')}")


if __name__ == "__main__":
    main()
