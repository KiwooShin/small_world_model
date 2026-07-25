# Lab 06 — LeWM, implemented by you

Implement LeWorldModel ([arXiv:2603.19312](https://arxiv.org/abs/2603.19312), Maes, Le
Lidec, Scieur, LeCun, Balestriero 2026) from a skeleton: the infrastructure is provided,
the four ideas that make it work are blanked out. When it runs, you'll have a ~10M-param
world model trained end-to-end from pixels with two loss terms, planning to image goals
with CEM — and an ablation switch that shows exactly why the second loss term exists.

This lab is the on-ramp to the repo's **LeWM vs DINO-WM on MuJoCo manipulation**
project ([roadmap M3/M4](../../docs/03-roadmap.md)): same model, real robot data.

## Why this paper

Every JEPA before LeWM needed a trick to avoid collapse — a frozen pretrained encoder
(DINO-WM), a stop-gradient, or an EMA teacher (V-JEPA). LeWM removes all of them:

> "We do not employ stop-gradient, exponential moving averages, or additional
> stabilization heuristics. Gradients are propagated through all components of the loss,
> and all parameters are optimized jointly in an end-to-end manner."

The replacement is a single statistical regularizer (SIGReg, from LeJEPA
[arXiv:2511.08544](https://arxiv.org/abs/2511.08544)) that pulls the batch of embeddings
toward N(0, I). A collapsed embedding distribution is maximally non-Gaussian, so the
degenerate minima of the prediction loss stop being minima of the total loss. Results:
beats DINO-WM on Push-T (96% vs 74% under the stable-worldmodel 50-step protocol) with
no pretrained features and plans up to 48× faster, because the whole state is one 192-dim
vector instead of a 256-patch grid.

## The objective you are building

With encoder `enc`, predictor `pred`, per-frame latents z_t = enc(o_t):

```
L_pred   = || pred(z_{0:t}, a_{0:t}) − z_{t+1} ||²     teacher-forced, target NOT detached
SIGReg   = mean over M random unit directions u of EP( {u·z_i} batch projected to 1-D )
EP(x)    = B ∫ |ecf_x(t) − e^{−t²/2}|² e^{−t²/2} dt    Epps–Pulley: empirical CF vs N(0,1)
L        = L_pred + λ · SIGReg        λ = 0.09 — the only tuned hyperparameter
```

## Files and tasks

| File | Status | What |
|---|---|---|
| [data.py](data.py) | provided | episode format, PushWorld collector |
| [sigreg.py](sigreg.py) | **TASK 1** | the Epps–Pulley sketch test — the paper's heart |
| [model.py](model.py) | **TASK 2** | AdaLN-zero action conditioning (rest of model provided) |
| [train.py](train.py) | **TASK 3** | the two-term loss assembly (loop provided) |
| [planner.py](planner.py) | **TASK 4** | CEM in latent space (eval harness provided) |
| [check.py](check.py) | provided | per-task self-checks with diagnostic messages |

Do them in order — each check is independent, so you always know which piece is wrong:

```bash
python -m labs.lab06_lewm.check --task 1     # after sigreg.py
python -m labs.lab06_lewm.check              # all
python -m labs.lab06_lewm.train              # ~minutes on the Spark
python -m labs.lab06_lewm.train --lambd 0.0 --tag collapse --epochs 8   # the ablation
python -m labs.lab06_lewm.planner            # goal-image planning eval
```

## What you should observe

- **Training (λ=0.09):** `pred` falls, `sigreg` stays low and roughly flat, `lat_std`
  hovers near 1, `eff_rank` in the tens. All four numbers print every epoch.
- **The ablation (λ=0):** `pred` falls *faster* — to near zero — while `lat_std`
  crashes. The model "solves" prediction by destroying the representation. This is
  collapse, live. (Complete collapse is the dramatic mode; watch `eff_rank` too —
  dimensional collapse can hide behind a healthy `lat_std`.)
- **Planning:** well-trained model ⇒ majority of goal-reaching episodes succeed;
  untrained or collapsed model ⇒ chance level. If checks pass but success is low,
  train longer or increase CEM iterations — planning quality tracks `pred` closely.

## Paper fidelity vs lab scale

Faithful: latent dim 192 (CLS token → BN-MLP projector), AdaLN-zero action conditioning,
causal predictor over a 3-frame history, teacher forcing with undetached targets,
SIGReg per time step (M=1024 projections, 17 trapezoid knots on [0,3], Gaussian window),
λ=0.09, AdamW + warmup-cosine, grad clip 1.0.

Scaled for a minutes-not-hours lab (official value in parens): image 64² (224²),
ViT depth 6 (12), patch 8 (14), predictor heads 8×32 (16×64), predictor MLP 768 (2048),
frame-skip 1 (5, with 5-step action blocks — restore this for the MuJoCo stage),
lr 1e-4 @ 20 epochs (5e-5 @ 100), CEM 256×10 iters (300×30). None of these change what
you implement.

Two things the official code does that this lab deliberately keeps: BatchNorm (not
LayerNorm) ending the projector — the paper notes a final LayerNorm "prevents our
anti-collapse objective from being optimized effectively" because it pins embeddings to
a sphere that can't match N(0, I) — and fresh random projection directions every SIGReg
call, which is what makes the sketch un-gameable (LeJEPA Fig. 7).

## References

- [LeWM paper](https://arxiv.org/abs/2603.19312) · [project page](https://le-wm.github.io/) · [official code](https://github.com/lucas-maes/le-wm) · [checkpoints/data](https://hf.co/collections/quentinll/lewm)
- [LeJEPA / SIGReg](https://arxiv.org/abs/2511.08544) · [code](https://github.com/rbalestr-lab/lejepa)
- [stable-worldmodel](https://github.com/galilai-group/stable-worldmodel) — the benchmark platform for the DINO-WM comparison stage
- [DiT / AdaLN-zero](https://arxiv.org/abs/2212.09748) — where the conditioning mechanism comes from
- Consult the official implementations only *after* your checks pass — reading the
  answer first defeats the lab.
