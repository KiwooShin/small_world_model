# lewm — LeWorldModel on MuJoCo, from scratch

A faithful, small-scale implementation of LeWM
([arXiv:2603.19312](https://arxiv.org/abs/2603.19312), Maes, Le Lidec, Scieur, LeCun,
Balestriero 2026): a JEPA world model trained **end-to-end from pixels** with two loss
terms — next-embedding prediction plus SIGReg ([LeJEPA,
arXiv:2511.08544](https://arxiv.org/abs/2511.08544)) — no frozen backbone, no
stop-gradient, no EMA teacher. Planning is CEM over latent rollouts toward a goal
*image*. Each frame is one 192-dim vector; the whole model is ~9M params.

## Run it

```bash
python -m lewm.collect                     # 500 reacher episodes  (~3 min)
python -m lewm.train                       # 20 epochs             (~15 min on GB10)
python -m lewm.eval                        # planning scores + GIF (~2 min)
python -m lewm.train --lambd 0 --tag collapse --epochs 8   # the ablation
```

## What the numbers mean

**Training dashboard** (printed per epoch, plotted to `media/lewm_<tag>_curves.png`):

| metric | healthy | collapsed (`--lambd 0`) |
|---|---|---|
| `pred` | falls steadily (~0.07 after 6 ep on the 2D toy) | falls to ~0.003 — *suspiciously* good |
| `sigreg` | settles ~3 | ~50 |
| `lat_std` | ~1 | ~0.003 — the tell |
| `eff_rank` | tens | unreliable once `lat_std` dies (rank of noise) |

The λ=0 run is the paper's core claim inverted: without SIGReg the model "solves"
prediction by destroying the representation. Same code, one flag.

**Eval scores** (`python -m lewm.eval`): success rate (fingertip within 5 cm of the
goal-image fingertip, 23 cm reach), mean/median final distance, and
`media/lewm_<tag>_plan.gif` showing goal | MPC execution side by side. The planner sees
only pixels; scoring uses true sim state.

## Files

| File | What |
|---|---|
| [envs/reacher.py](envs/reacher.py) | 2-DoF planar reacher (MJCF inline, EGL headless, 64², state get/set, scripted PD-waypoint policy) |
| [sigreg.py](sigreg.py) | the Epps–Pulley sketch test — the anti-collapse mechanism, fully commented |
| [model.py](model.py) | ViT-Tiny encoder → CLS → BN-projector → 192-d latent; AdaLN-zero causal predictor |
| [data.py](data.py) / [collect.py](collect.py) | npz episode format shared by all envs, offline collection |
| [train.py](train.py) | two-term loss, warmup-cosine AdamW, collapse dashboard, curve plots |
| [planner.py](planner.py) / [eval.py](eval.py) | CEM in latent space; teleport-based goal-image protocol with scores |

## Fidelity vs paper

Faithful: 192-d CLS latent, BatchNorm projector (a final LayerNorm would pin embeddings
to a sphere SIGReg can't Gaussianize), AdaLN-zero action conditioning, 3-frame history,
teacher forcing with **undetached** targets, SIGReg per time step (1024 fresh projections
per call, 17 trapezoid knots on [0,3]), λ=0.09, AdamW + 1% warmup cosine, grad clip 1.0,
seed 3072.

Scaled (official in parens): 64² images (224²), ViT depth 6 patch 8 (12, patch 14),
predictor heads 8×32 mlp 768 (16×64, 2048), action repeat 5 in the env instead of
frame-skip-5 action blocks in the data (blocks return for the manipulation tasks),
lr 1e-4 @ 20 epochs (5e-5 @ 100), CEM 256×10 (300×30).

## Next (roadmap M2/M3)

Planar pushing scene (first contact dynamics) → Franka/Unitree assembly assets →
DINO-WM comparison under the identical protocol. See
[docs/03-roadmap.md](../docs/03-roadmap.md).
