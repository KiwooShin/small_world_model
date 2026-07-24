# The world-model landscape, mid-2026

*Written 2026-07-24. Facts about post-2025 work were checked against sources at the bottom;
older material is standard background. Anything here will rot — re-check before trusting a
specific number.*

## 0. Why the term is confusing

"World model" is used for at least four different research programs that share a slogan and
disagree about almost everything else. A July 2026 perspective paper opens by conceding there
is "no consensus on what a world model fundamentally is, what it should predict, or how it
should be built" ([Definition and Roadmap, arXiv:2607.06401][roadmap]). So the first job is not
to pick a favorite, but to learn the axes people are actually disagreeing about.

The useful common definition: **a learned model that predicts how an environment evolves,
conditioned on actions.** Drop "conditioned on actions" and you have a video generator. Drop
"learned" and you have a simulator. The interesting systems sit exactly on that boundary.

## 1. The axes that matter

Before the family tree, the questions that actually separate the systems:

| Axis | Options | Why it matters |
|---|---|---|
| **Prediction target** | pixels · discrete tokens · stochastic latent · abstract embedding | Determines what the loss can even see. Pixel losses spend capacity on texture; embedding losses can collapse. |
| **Action conditioning** | none · discrete · continuous · latent-inferred | Action labels are the scarcest resource in the field. Latent-action methods exist to avoid needing them. |
| **Dynamics parameterization** | RNN/SSM · transformer · diffusion/flow | Sets rollout cost. Recurrent = O(1) per step; transformer = O(context); diffusion = O(denoise steps). |
| **Horizon & memory** | seconds · a minute · unbounded | The current wall. Most systems drift or forget within ~1 min. |
| **What it is *for*** | policy learning · planning/MPC · simulation & data-gen · evaluation | A model good enough to plan through can be far worse than one good enough to look at. |

That last row is the one most often skipped and it drives everything. A model used for
imagination-based RL needs *reward and value* to be right and can tolerate blurry pixels. A
model used as a game engine needs the opposite. They are not the same artifact.

## 2. Four families

### A. Latent state-space models (compact, recurrent, sample-efficient)

The oldest continuous line. Encode observations to a small latent, roll a recurrent state
forward, decode only for supervision. Ha & Schmidhuber's *World Models* (2018) set the V–M–C
template (vision / memory / controller); PlaNet (2019) added the RSSM — a state split into a
deterministic path and a stochastic path, which is the single most reused architectural idea in
this family; Dreamer V1→V3 turned it into a general agent, with DreamerV3 notably hitting a
single hyperparameter set across domains and collecting diamonds in Minecraft from scratch.

The 2025/2026 turn: **Dreamer 4** pivots this line to the *offline* setting — a 2B-parameter
agent using an efficient transformer trained with a "shortcut forcing" objective (flow-matching
based, but predicting the clean state, x-prediction, rather than the update vector), simulating
game mechanics at ~21 FPS on a single GPU, and reported as the first to solve Minecraft's
obtain-diamonds task purely from a fixed offline dataset ([summary][dreamer4]).

Note what happened there: the "recurrent latent" family adopted a transformer *and* a
generative-diffusion-family objective. The families are converging.

**Strength:** compute efficiency, sample efficiency, and it produces a *policy*, not just video.
**Weakness:** small latents throw away detail; historically domain-specific rather than
general-purpose.

### B. Tokenized autoregressive transformers (discrete, scalable, inspectable)

Quantize frames into discrete tokens (VQ-VAE and successors), then model the sequence of
(frame tokens, action) autoregressively with a transformer. IRIS (*Transformers are
Sample-Efficient World Models*, 2022) is the clean reference implementation of the idea; TWM is
a sibling. Genie (2024) added the important trick for scale: **latent actions**, inferred
unsupervised from unlabeled video, which sidesteps the action-label bottleneck entirely.

Genie 3 is the current flagship of the "playable world" framing: interactive, navigable
environments generated in real time from a text prompt at 720p / 24 fps, with visual memory
holding consistency for about a minute; announced Aug 2025, publicly launched Jan 29 2026 for
Google AI Ultra subscribers in the US. Its stated limits are instructive — unreadable in-scene
text, difficulty with multiple autonomous agents, and interaction that is essentially navigation
rather than manipulation ([DeepMind][genie3], [overview][genie3blog]). Waymo built a driving-specific
variant, the Waymo World Model, in Feb 2026.

**Strength:** scales like language modeling; discrete tokens are easy to inspect and to condition.
**Weakness:** tokenizer is a hard information bottleneck; rollout cost grows with context.

### C. Diffusion / flow generative world models (highest fidelity)

Model the next frame (or latent chunk) with a diffusion or flow model conditioned on history
and action. GameNGen (2024) showed a diffusion model can serve as a playable DOOM engine;
DIAMOND showed diffusion dynamics beat discrete-token dynamics on Atari at matched budget,
because the tokenizer was destroying task-relevant detail. NVIDIA's Cosmos line productized
this for physical AI (Cosmos 3 released 2026-05-31).

2026's theme here is **long, cheap, controllable rollouts**. NVIDIA's SANA-WM is a 2.6B
diffusion-transformer trained natively for one-minute 720p generation with metric-scale 6-DoF
camera control, shipped in three single-GPU variants — bidirectional, chunk-causal
autoregressive, and a few-step distilled autoregressive one that denoises 60 s of 720p in ~34 s
on a single RTX 5090 with NVFP4 ([writeup][sanawm]). Tencent's HY-World 2.0 (2026-04-15) is a
~34 GB open release runnable on a single 24 GB card.

**Strength:** fidelity, and increasingly *speed* via distillation.
**Weakness:** expensive per step; and looking right is not the same as being right — see §4.

### D. Joint-embedding predictive (JEPA): predict representations, not pixels

The dissenting family. Never reconstruct: encode observation and target, and predict the
*target's embedding* from the current embedding plus action. The bet is that pixel prediction
wastes capacity on unpredictable detail, and that control only needs the predictable part.

The empirical case has firmed up considerably:
- **V-JEPA 2-AC** post-trains an action-conditioned world model on under 62 hours of unlabeled
  robot video and does zero-shot pick-and-place on a Franka arm using image goals ([arXiv:2506.09985][vjepa2]).
- **DINO-WM** — a JEPA on top of a *frozen* DINOv2 encoder — outperforms DreamerV3 and TD-MPC2
  on goal-conditioned planning when no reward is available.
- **LeWM** (2026) is an end-to-end action-conditioned JEPA trained from raw pixels with only two
  loss terms, specifically targeting the representation-collapse pathologies that made
  end-to-end latent prediction fragile ([discussion][lewm]).
- Systematic studies now exist on *what actually drives* JEPA planning success
  ([arXiv:2512.24497][jepadrivers]), plus early generalization theory ([arXiv:2606.27014][jepatheory]).

**Strength:** cheap, sample-efficient, aimed directly at control.
**Weakness:** collapse risk; you cannot look at the prediction, which makes debugging and
evaluation genuinely harder.

## 3. The live debate

**Does generating video mean understanding the world?** The generative camp (B, C) argues
scale plus pixels gets you a general simulator. The JEPA camp (D) argues you are burning
compute on texture and calling it physics. The 2026 evidence is mixed on purpose: video models
produce gorgeous rollouts that violate conservation of mass, while latent models plan well and
can't show you what they think.

A useful reframing from the **World Action Models** survey (2026-06-18): these are not "video
generators with an action head." The design choices trade representational richness against
compute, memory, latency and action-label cost — and the field is trending toward *generating
less of the future while preserving what control requires* ([arXiv:2606.20781][wamsurvey]).
That sentence is the best one-line summary of where 2026 sits.

## 4. Evaluation is the field's weakest link

Consistently named as an open problem: fragmented standards, no agreed protocol
([survey][survey]). What exists now clusters into:

- **Physics / commonsense probes:** PhyGenBench, WorldModelBench, PhyWorldBench, PhyGround,
  LikePhys. The recurring failure modes are concrete — objects floating unsupported, fluids
  losing mass, shadows inconsistent with the light source.
- **Long-horizon state:** MBench, WorldPrediction, WorldReasonBench — do persistent states
  survive, can the model support procedural planning. A 2026 paper argues bluntly that current
  world models **lack a persistent state core** ([arXiv:2606.20545][statecore]).
- **Functional utility:** WorldArena, and the honest test — can a policy trained or planned
  inside the model actually act in the real environment.

Treat generative metrics (FVD and friends) as necessary-not-sufficient. The functional test is
the one that matters, and it is the one this repo's lab 07 targets.

## 5. Where to get code

- [`galilai-group/stable-worldmodel`][stablewm] — reproducible platform; reference DINO-WM,
  LeWM, PLDM, plus GCBC/GCIVL/GCIQL baselines; 30+ envs (DMC, Gymnasium, OGBench, Craftax, ALE,
  PushT); planners CEM / iCEM / MPPI / predictive sampling / gradient / augmented-Lagrangian.
  The most directly useful thing for this repo's labs 06–07.
- [`NVIDIA/cosmos`][cosmos] — open platform of world models + datasets for physical AI.
- [`OpenDCAI/OpenWorldLib`][openworldlib] — unified codebase across world-model families.
- Reading lists: [`JiahuaDong/Awesome-World-Models`][awesome1],
  [`OpenMOSS/Awesome-WAM`][awesome2] (world action models),
  [`ziqihuangg/Awesome-From-Video-Generation-to-World-Model`][awesome3],
  [`opendilab/awesome-model-based-RL`][awesome4].

## 6. Surveys, if you want one authoritative sweep

- [*A Definition and Roadmap for World Models*][roadmap] (2026-07) — 58-page perspective;
  read for the definitional argument and the staged roadmap.
- [*World Models: A Comprehensive Survey*][survey] (2026-05-28) — taxonomy across architecture
  (representation format, dynamics formulation, input modality, learning paradigm, application);
  five method families (state-space/recurrent, transformer, diffusion, physics-informed,
  language-augmented multimodal); four reasoning paradigms (imagination-based planning, latent
  policy learning, counterfactual reasoning, planning under uncertainty).
- [*World Action Models: A Survey*][wamsurvey] (2026-06-18) — the control-centric view.
- [*World Model for Robot Learning: A Comprehensive Survey*][robotsurvey] (2026-04-30) — if the
  robotics angle is the priority.

---

### Sources

[roadmap]: https://arxiv.org/pdf/2607.06401
[survey]: https://arxiv.org/abs/2606.00133
[wamsurvey]: https://arxiv.org/abs/2606.20781
[robotsurvey]: https://arxiv.org/abs/2605.00080
[dreamer4]: https://arxiviq.substack.com/p/dreamer-4-training-agents-inside
[genie3]: https://deepmind.google/blog/genie-3-a-new-frontier-for-world-models/
[genie3blog]: https://wavespeed.ai/blog/posts/google-deepmind-genie-3-world-model-2026/
[sanawm]: https://www.marktechpost.com/2026/05/16/nvidia-introduces-sana-wm-a-2-6b-parameter-open-source-world-model-that-generates-minute-scale-720p-video-on-a-single-gpu/
[vjepa2]: https://arxiv.org/abs/2506.09985
[jepadrivers]: https://arxiv.org/abs/2512.24497
[jepatheory]: https://arxiv.org/html/2606.27014v1
[lewm]: https://medium.com/@adnanmasood/leworldmodel-and-the-case-for-stable-latent-world-models-0e4c33ca0f3c
[statecore]: https://arxiv.org/pdf/2606.20545
[iris]: https://arxiv.org/pdf/2209.00588
[stablewm]: https://github.com/galilai-group/stable-worldmodel
[cosmos]: https://github.com/nvidia/cosmos
[openworldlib]: https://github.com/OpenDCAI/OpenWorldLib
[awesome1]: https://github.com/JiahuaDong/Awesome-World-Models
[awesome2]: https://github.com/OpenMOSS/Awesome-WAM
[awesome3]: https://github.com/ziqihuangg/Awesome-From-Video-Generation-to-World-Model
[awesome4]: https://github.com/opendilab/awesome-model-based-RL
