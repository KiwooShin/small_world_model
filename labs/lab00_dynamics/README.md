# Lab 00 — action-conditioned pixel dynamics (the naive baseline)

**The idea:** the most literal world model possible. One conv net,
`f(last 2 frames, action) → next frame`, trained with MSE. No latent space, no
stochasticity, no recurrence. ~1.1M parameters.

**Why start here:** every architectural idea in the field is a repair for a failure this
model exhibits. You need to *see* the failures first:

1. **One-step prediction is easy.** After a few epochs the one-step output is near-perfect.
   This is the trap — one-step metrics say the model is done.
2. **Autoregressive rollout drifts.** Feed predictions back in and small errors compound:
   the ball blurs, then smears, then physics quietly dies. The model was never trained on
   its own outputs (covariate shift), and MSE averages over uncertain futures (blur).

**The demo:** `media/lab00_rollout.gif` shows ground truth | imagination | |error| side by
side; the imagination panel gets a red border the moment the model starts eating its own
predictions. `media/lab00_psnr.png` quantifies the same thing as PSNR vs horizon.

```bash
python -m labs.lab00_dynamics.run            # ~5 min on a GB10
python -m labs.lab00_dynamics.run --smoke    # 1-minute sanity check
```

**What lab 01 adds:** predict in a learned latent space instead of pixel space — same data,
same budget — and watch the coherent horizon lengthen.

## Files

- [env.py](env.py) — PushWorld: exact 2D physics (pusher + bouncing ball), numpy only
- [data.py](data.py) — scripted ball-seeking policy, in-memory transition dataset
- [model.py](model.py) — UNet-lite with FiLM action conditioning
- [run.py](run.py) — train → rollout → GIF + PSNR curve
