# Roadmap

Goal of the project: demonstrate, with working code and demo videos, four things —
**(1)** studied understanding of world models, **(2)** ability to build and train them,
**(3)** application to robotics, **(4)** serious treatment of efficiency (everything runs on
one DGX Spark). Every milestone ends in a fancy, self-explanatory demo GIF in the README.

The recipes and feasibility numbers behind these choices are in
[02-small-wm-frontier-2026.md](02-small-wm-frontier-2026.md).

## M0 — Study foundation + lab ladder begins ✅ (started 2026-07-24)

- Landscape map, reading path, notes system ([docs/](.), [notes/](../notes/))
- Lab 00 running: naive pixel dynamics, compounding-error demo
- **Demo:** GT vs imagination vs error GIF; PSNR-vs-horizon curve (39 dB → 21 dB over 60 steps)

## M1 — The core ideas, from scratch (labs 01–03)

Latent prediction → RSSM with stochastic state → actor-critic trained purely in imagination,
all on PushWorld, all in minutes-to-an-hour per run.

- **Demo:** one context, N sampled imagined futures fanning out (the "stochastic latent"
  money shot); agent solving PushWorld having never trained in the real env, with a
  real-env return curve vs a model-free baseline.
- Study phases 0–1 complete (Ha & Schmidhuber → DreamerV3), notes written.

## M2 — Architecture bake-off (labs 04–05) + MuJoCo data engine

- Tokens+transformer and few-step diffusion dynamics on the same env/budget; the
  fidelity-vs-steps grid (1/2/4/8 denoise steps) is the efficiency demo.
- In parallel: a MuJoCo data engine reusing the existing Unitree/Franka assembly assets —
  scripted policies dumping unlimited (RGB, proprio, action) trajectories, native on the
  Spark. This is the dataset for M3–M5.
- **Demo:** four families rolling out the same trajectory side by side; MuJoCo dataset
  montage.

## M3 — Robotics payoff #1: plan with a world model (lab 06 at scale)

DINO-WM recipe on our own MuJoCo manipulation data: frozen DINOv2, small ViT dynamics
head, CEM/MPC in latent space, goal-image conditioning. stable-worldmodel as reference,
implementation ours.

- **Demo:** robot arm reaching/pushing to a goal *image* — planned entirely inside the
  world model, executed in MuJoCo; side panel shows the model's imagined plan vs execution.

## M4 — Robotics payoff #2: V-JEPA 2-AC-style post-training (flagship)

The strongest compute-for-result recipe of 2026, replicated on our data: frozen video
encoder (V-JEPA 2 ViT-L or DINOv2), block-causal action-conditioned predictor
(~100–300M — sized to what the Spark trains in days), teacher forcing + rollout loss, CEM
planning. Meta needed <62 h of robot video; our MuJoCo engine generates that in a weekend.

- **Demo:** zero-shot manipulation in *held-out* MuJoCo scenes via latent planning;
  comparison table vs M3 (success rate, planning time per action).

## M5 — Efficiency + generative story: LoRA a pretrained WM

LoRA Cosmos-Predict2.5-2B (official ~50M-param recipe; ≈3–4 Spark-days; needs the
documented aarch64 patches — SDPA instead of flash-attn/TE) or SANA-WM 2.6B on our robot
videos → photoreal "dreams" of the Unitree/Franka. Optionally close the loop
WorldEval-style: evaluate an M3/M4 policy inside the finetuned generative model.

- **Demo:** the fancy one — photoreal imagined robot futures next to the MuJoCo ground
  truth; plus an honest efficiency table (params trained, wall-clock, memory, FPS) across
  M1→M5.

## M6 (stretch) — Evaluation dashboard (lab 07)

Drift, memory persistence, physics probes, and the functional test (policy/plan success)
applied uniformly to every model built in M1–M5. One page, one artifact.

---

### Standing constraints (locked)

- Single DGX Spark; BF16 (+FP8) training; SDPA everywhere; no flash-attn/TE/xformers.
- MuJoCo for data (native aarch64). Unreal Engine only if a second x86 machine appears;
  photorealism via transfer models otherwise.
- Every milestone: README GIF + a short "what this shows" caption. Small models,
  full understanding — no cluster jobs.
