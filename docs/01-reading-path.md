# Reading path

Ordered so that each phase's papers are exactly what the matching lab implements. Read the
papers of a phase, build the lab, write a note per paper in [`notes/`](../notes/) using the
[template](../notes/TEMPLATE.md). Don't read ahead of what you've built — the papers all make
more sense with a training curve of your own to compare against.

Each phase ends with **checkpoints**: questions you should be able to answer from memory,
whiteboard-style. If you can't, the phase isn't done.

---

## Phase 0 — The core loop (→ labs 00–01)

1. **Ha & Schmidhuber, *World Models* (2018)** — [arXiv:1803.10122](https://arxiv.org/abs/1803.10122).
   The V–M–C decomposition. Read for the framing, not the specific architecture.
2. **Interactive companion** — [worldmodels.github.io](https://worldmodels.github.io/).

**Checkpoints:** Why does predicting in latent space beat predicting in pixel space, and what
is the failure mode of each? Why does a model trained one-step-ahead fall apart when rolled
out autoregressively (covariate shift / compounding error), and what are three mitigations?

## Phase 1 — Stochastic latents and imagination (→ labs 02–03)

1. **PlaNet** (Hafner et al. 2019) — [arXiv:1811.04551](https://arxiv.org/abs/1811.04551). The RSSM. The one architecture to know cold.
2. **DreamerV1** (2020) — [arXiv:1912.01603](https://arxiv.org/abs/1912.01603). Actor-critic in imagination, backprop through the model.
3. **DreamerV2** (2021) — [arXiv:2010.02193](https://arxiv.org/abs/2010.02193). Discrete latents + KL balancing. Read carefully — these two tricks carry the family.
4. **DreamerV3** (2023) — [arXiv:2301.04104](https://arxiv.org/abs/2301.04104). symlog, two-hot returns, one hyperparameter set for everything. Also note the *parameter-scaling* results — a "small world model" data point.

**Checkpoints:** Draw the RSSM (deterministic path, stochastic path, prior vs posterior).
Why is the KL balanced between prior→posterior and posterior→prior? Why can a *worse*
video-prediction model be a *better* model for RL? Why train the policy in imagination instead
of on real replay?

## Phase 2 — Tokens and transformers (→ lab 04)

1. **VQ-VAE** (2017) — [arXiv:1711.00937](https://arxiv.org/abs/1711.00937). The tokenizer.
2. **IRIS** (2022) — [arXiv:2209.00588](https://arxiv.org/abs/2209.00588). World model as language model over frame tokens. The cleanest codebase-paper pair in the field.
3. **Genie** (2024) — [arXiv:2402.15391](https://arxiv.org/abs/2402.15391). Latent actions from unlabeled video — the trick that unlocks internet-scale data.
4. Skim: Genie 3 announcement (2025/2026) for where this line went as a product.

**Checkpoints:** What information does a VQ tokenizer destroy and when does that matter for
control (this is DIAMOND's whole argument)? How does the latent-action model recover actions
without labels, and what stops it from collapsing to a constant?

## Phase 3 — Diffusion dynamics (→ lab 05)

1. **DDPM** (2020) — [arXiv:2006.11239](https://arxiv.org/abs/2006.11239) — plus a flow-matching primer; you need both vocabularies.
2. **DIAMOND** (2024) — [arXiv:2405.12399](https://arxiv.org/abs/2405.12399). Diffusion world model beats tokens on Atari at matched compute.
3. **Dreamer 4** (2025) — shortcut forcing: flow-matching-style objective, x-prediction, few-step rollout at ~21 FPS on one GPU. The current reference point for *efficient* high-fidelity dynamics.

**Checkpoints:** Why does v-prediction vs x-prediction matter for a world model specifically
(error accumulation under autoregressive rollout)? How do shortcut/consistency models get away
with 1–4 denoising steps? What does diffusion buy over tokens, and at what rollout cost?

## Phase 4 — JEPA: don't reconstruct (→ lab 06)

1. **I-JEPA** (2023) — [arXiv:2301.08243](https://arxiv.org/abs/2301.08243). The architecture and the collapse problem.
2. **DINO-WM** (2024) — [arXiv:2411.04983](https://arxiv.org/abs/2411.04983). Frozen DINOv2 + small dynamics head + MPC. *The* recipe for small-compute world modeling.
3. **V-JEPA 2 / V-JEPA 2-AC** (2025) — [arXiv:2506.09985](https://arxiv.org/abs/2506.09985). Action-conditioned head post-trained on <62 h of robot video → zero-shot Franka pick-and-place.
4. **What drives success in physical planning with JEPA world models?** (2025/26) — [arXiv:2512.24497](https://arxiv.org/abs/2512.24497). The ablation study to calibrate your intuitions.

**Checkpoints:** Name three collapse-prevention mechanisms and which one each JEPA system
uses. Why does planning (MPC/CEM) in embedding space work without a decoder? What breaks JEPA
planning at long horizon and what are the current fixes?

## Phase 5 — Robotics use + honest evaluation (→ lab 07)

1. **World Model for Robot Learning survey** (2026) — [arXiv:2605.00080](https://arxiv.org/abs/2605.00080). The application map: policy learning, planning, neural sim, data-gen, evaluation.
2. **World Action Models survey** (2026) — [arXiv:2606.20781](https://arxiv.org/abs/2606.20781). The control-centric reframing.
3. Evaluation papers: WorldModelBench + one physics probe (e.g. [LikePhys](https://arxiv.org/pdf/2510.11512)) + the persistent-state critique ([arXiv:2606.20545](https://arxiv.org/pdf/2606.20545)).
4. **A Definition and Roadmap for World Models** (2026-07) — [arXiv:2607.06401](https://arxiv.org/pdf/2607.06401). Read *last*; it's a synthesis and reads best when you can argue with it.

**Checkpoints:** Given a robotics task and one GPU, which world-model recipe do you pick and
why — and what evaluation would convince a skeptic it worked? What are the four distinct ways
a world model can earn its keep in a robot learning pipeline?

---

*See [02-small-wm-frontier-2026.md](02-small-wm-frontier-2026.md) for the frontier-lab and
efficiency-focused research sweep (what the labs are doing and what's trainable on one GPU).*
