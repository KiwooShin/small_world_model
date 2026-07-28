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
