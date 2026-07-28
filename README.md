# small_world_model

World models, learned from first principles and built small: everything in this repo is
implemented from scratch, heavily commented, and trained on a single NVIDIA DGX Spark.
The centerpiece is a from-scratch **LeWorldModel** (LeWM, [arXiv:2603.19312](https://arxiv.org/abs/2603.19312))
trained on MuJoCo robot tasks and evaluated by goal-image planning — with a demo artifact
for every milestone.

**Why LeWM?** The strongest compute-per-result recipes of 2025–26 are small: LeWM trains a
stable pixel JEPA end-to-end at ~15M params in hours on one GPU — no frozen backbone, no
stop-gradient, no EMA teacher; a single statistical regularizer (SIGReg) replaces all of
them — and it beats DINO-WM on Push-T while planning up to 48× faster. Small is not the
compromise — it's the interesting regime. The full argument, with sources:
[docs/02-small-wm-frontier-2026.md](docs/02-small-wm-frontier-2026.md).

## Milestones

| | Milestone |
|---|---|
| ✅ | **M0** — study foundation: field map, reading path, research sweep ([docs/](docs/)) |
| 🔨 | **M1** — LeWM from scratch on a simple MuJoCo task ([lewm/](lewm/)): train, plan, score |
| ⬜ | **M2** — scale the task ladder: reacher → pushing → Franka/Unitree assembly scenes |
| ⬜ | **M3** — the open question: LeWM vs DINO-WM on contact-rich manipulation, same data, same CEM protocol |
| ⬜ | **M4** — V-JEPA 2-AC-style post-training on our own robot data |
| ⬜ | **M5** — LoRA a pretrained generative WM → photoreal robot dreams |

Full plan: [docs/03-roadmap.md](docs/03-roadmap.md). (An earlier incremental "labs ladder"
— including the pixel-dynamics compounding-error demo — lives in git history before
2026-07-28; its verified LeWM components became [lewm/](lewm/).)

## The model in one diagram

```
frames (B,T,3,64,64) ──ViT-Tiny──► CLS ──MLP+BN──► z_t ∈ R^192      (one token per frame)
actions (B,T,A)      ──MLP──────────────────────► a_t ∈ R^192
(z, a) history ──causal transformer, AdaLN-zero action conditioning──► ẑ_{t+1}

L = ||ẑ_{t+1} − z_{t+1}||²  +  λ · SIGReg(z)          λ = 0.09, targets NOT detached
        prediction              anti-collapse: Epps–Pulley test of z against N(0,I)
                                on random 1-D projections (sketched Cramér–Wold)

Planning: encode goal image once, CEM over action sequences through latent rollouts,
cost = MSE to goal latent at the last step. No decoder anywhere.
```

## Results: the head-to-head (updated 2026-07-28, campaign day 1)

Goal-image planning on the MuJoCo reacher: 25 episodes, success = fingertip
within 5 cm of the goal-image pose (first passage), mean start distance
0.19 m. All models trained on the same 2000-episode / 300k-frame offline
dataset (LeWM: 60 epochs fs=5, prediction MSE 0.004, effective rank 48).

| Reacher | success | mean final dist | state |
|---|---|---|---|
| **DINO-WM baseline** (frozen DINOv2, patch grid) | **56%** | **0.076 m** | 49×384 patch tokens |
| LeWM + probed-point cost *(diagnostic†)* | 28% | 0.085 m | 192-d token + linear readout |
| LeWM + GoalMSE (the paper's cost) | 12% | 0.179 m | 192-d token |
| collapsed control (λ=0) / zero / random | 0% | ≈ start | — |

† the probe needs offline state labels, so it is an upper bound, not pure
goal-image planning.

**The finding.** LeWM's latent *contains* the state — a linear probe reads
the fingertip position to 1.5 cm median error — but **GoalMSE cannot see
it**: SIGReg whitens the space, latent distance decorrelates from task
distance beyond ~5 cm (corr 0.35 → ~0.0), and the planner gets no gradient
toward distant goals. Identical machinery with a probed-point cost halves
the final distance (0.179 → 0.085 m). DINO-WM's spatially-structured patch
grid is the natural fix — feature-MSE over patches preserves *where things
are* — and wins outright at this scale. Chain of evidence that localized
this (each step committed with its pipeline): honest-eval rebuild with
chance baselines → 5-config CEM sweep (flat: planner exonerated) →
open-loop rollout drift measurement (13% of random-pair distance at 8
steps: dynamics exonerated) → RSA + linear probe (information present,
metric blind) → probe-cost intervention (28%) → DINO-WM baseline (56%).

Caveat honestly noted: this is lab scale (64², ~9M params, one seed);
the LeWM paper reports the opposite ordering on Push-T at 224² with ~45×
our original training budget. "At what scale does end-to-end overtake
frozen-pretrained?" is now this repo's driving question. Contact-rich
pusher head-to-head is running.

## Repo layout

| Where | What |
|---|---|
| [`lewm/`](lewm/) | The LeWM implementation: model, SIGReg, MuJoCo envs, training, CEM planning/eval |
| [`docs/`](docs/) | Field map ([00](docs/00-landscape-2026.md)), reading path ([01](docs/01-reading-path.md)), frontier/efficiency research sweep ([02](docs/02-small-wm-frontier-2026.md)), roadmap ([03](docs/03-roadmap.md)) |
| [`notes/`](notes/) | Per-paper study notes ([template](notes/TEMPLATE.md)) |
| [`media/`](media/) | Demo artifacts, one set per milestone |

## Quick start

```bash
pip install -r requirements.txt
python -m lewm.collect            # MuJoCo reacher episodes -> data/
python -m lewm.train              # train LeWM; per-epoch collapse dashboard
python -m lewm.train --lambd 0    # the ablation: watch collapse live
python -m lewm.eval               # CEM goal-image planning -> success rate
```

Developed against Python 3.13 / PyTorch 2.11+cu128 / MuJoCo 3.11 (EGL, headless) on an
NVIDIA GB10 (DGX Spark, 121 GB unified memory). Everything trains in minutes-to-hours;
that's the point of "small" in the repo name.

## Working notes

Paper notes go in [`notes/`](notes/), one file per paper, using
[notes/TEMPLATE.md](notes/TEMPLATE.md). The template asks for the thing that is easy to
skip and most worth writing down: *what would break if you removed this paper's one idea.*
