# Small & efficient world models — frontier sweep and what fits a DGX Spark

*Compiled 2026-07-24 from four parallel research sweeps (frontier labs; small/efficient
architectures; robotics finetuning; runnable codebases). Sources at the bottom of each
section. This doc exists to answer one question: **given one DGX Spark (GB10, 121 GB unified
memory, aarch64, ~273 GB/s bandwidth), what world-model work is actually worth doing?***

## TL;DR

1. **The field split into two camps, and the small-compute camp is winning on
   compute-per-result.** Generative pixel world models (Genie 3, Cosmos, SANA-WM, Runway
   GWM-1) are products trained on clusters. Latent-prediction world models for planning
   (JEPA line, Dreamer line) get robot-relevant results at 7M–300M trainable params on
   single GPUs. LeCun left Meta to bet a $1.03B seed on exactly the second camp (AMI Labs).
2. **The single best recipe for this hardware: frozen pretrained encoder + small trainable
   dynamics head.** DINO-WM (frozen DINOv2 + ViT predictor, beats DreamerV3 on planning),
   V-JEPA 2-AC (frozen 1B encoder + 300M action-conditioned predictor trained on <62 h of
   robot video → zero-shot Franka pick-and-place), LeWM (15M params, single GPU, hours).
3. **Finetuning big pretrained world models is feasible only via LoRA-class adapters**:
   NVIDIA's official Cosmos-Predict2.5-2B robot-video LoRA is ~50M trainable params /
   17 H100-hours (≈3–4 Spark-days). Full finetunes of 2B video models are 32×A100 jobs — out.
4. **Data comes from MuJoCo, not the real world and not Unreal on this box** — native
   aarch64 wheels, exact physics, free action labels, and existing Unitree/Franka assets.
   Photorealism, if wanted, is a *post-processing* step (Cosmos-Transfer-style sim2real
   restyling), not a rendering-engine decision.

---

## 1. Frontier-lab map (July 2026)

| Lab | Flagship | Predicts | Open? | One-line status |
|---|---|---|---|---|
| Meta FAIR / MSL | V-JEPA 2 / 2-AC (1B enc + 300M AC head) | embeddings | **weights+code (MIT)** | The robotics-relevant open release; V-JEPA 2.1 checkpoint refresh 2026, no V-JEPA 3 |
| **AMI Labs** (LeCun, ex-Meta Nov 2025) | LeJEPA (theory), **LeWM (15M params)** | embeddings | code | $1.03B seed (Mar 2026) explicitly to build small latent world models for industry/robotics |
| Google DeepMind | Genie 3 (Project Genie, GA Jan 2026), Dreamer 4 (~2B), SIMA 2 | pixels / latents | closed (Dreamer 4 paper-only) | Closed loop: SIMA 2 agent trains *inside* Genie 3 worlds; Waymo World Model built on Genie 3 |
| OpenAI | Sora 2 → **discontinued Apr 2026** | pixels | closed | Exited consumer world-sim; robotics team rebuilding, nothing published |
| NVIDIA | Cosmos 3 omnimodel (Nano 16B/Super 64B, actions native), SANA-WM 2.6B, GR00T-Dreams | pixels+actions | **open (OpenMDW)** | The open-weights leader; Cosmos 3 Nano runs on a DGX Spark in ~30 GB |
| World Labs (Fei-Fei Li) | Marble (GA Feb 2026), RTFM | 3D scenes / pixels | API only | $1.23B raised; 3D scene generation, not action-conditioned dynamics |
| Tencent Hunyuan | HY-WorldPlay (8B/5B, real-time interactive), HY-World 2.0 (3D) | pixels / 3D | **open** | WorldPlay-5B training code public — a realistic small-VRAM finetune target |
| Runway / Decart / Odyssey | GWM-1, Oasis 3, Odyssey-2/Agora-1 | pixels | closed/API | Real-time interactive video worlds as products; robotics branches nascent |
| Wayve / Waymo | GAIA-3 (15B), Waymo WM | latents/pixels | closed | Driving world models consolidated on synthetic-data + policy-eval use cases |
| Thinking Machines | — (Tinker, Inkling LLM) | — | — | Not a world-model lab |
| 1X / Skild / Physical Intelligence | 1X World Model (policy eval), Skild+Cosmos, π0.7 | pixels / actions | closed | Humanoid labs use WMs mainly to *evaluate* policies before execution |

Cross-cutting: action conditioning became table stakes in 2026 (camera pose, keyboard,
robot commands, driving controls); open-weights leadership = NVIDIA + Tencent; the money
moved to world models ($1B+ rounds at AMI, World Labs, Skild) while OpenAI left the race.

## 2. How small can a working world model be?

Parameter counts of published systems with real results — the existence proof that this
repo's premise ("small") is not a compromise:

| Model | Trainable params | Hardware | Result |
|---|---|---|---|
| Drama (ICLR'25) | **7M** (Mamba-2) | 1 GPU ("laptop-trainable") | ~105% mean HNS Atari-100k — matches 30M IRIS |
| STORM | ~6M transformer | ~4.3 h × 1 GPU/game | 126.7% HNS |
| DIAMOND | 13M (4.4M diffusion WM) | 12 GB VRAM, ~2.9 d/game RTX 4090 | 1.46 HNS, best pure-WM Atari '24 |
| **LeWM** (Mar 2026) | **15M**, 2 loss terms | 1 GPU, hours | Stable end-to-end pixel JEPA; plans 48× faster than foundation WMs |
| DreamerV3 | 12M–400M family | 1 GPU/run | Nature 2025; note: *larger is also more data-efficient* — small has a real cost |
| TWISTER (ICLR'25) | Dreamer-scale | 1 GPU | **162% HNS — top no-search WM agent as of mid-2026** |
| DINO-WM | small ViT head (frozen DINOv2) | 1-GPU-class | PushT 0.90 vs IRIS 0.32; beats DreamerV3/TD-MPC2 on reward-free planning |
| V-JEPA 2-AC | **300M head** (frozen 1B enc) | post-training only, <62 h video | zero-shot real-robot pick-and-place; plans in 16 s vs Cosmos's ~4 min |
| Dreamer 4 | ~2B | trained on 256–1024 TPUs; **runs** 21 FPS on 1 GPU | Minecraft diamonds, offline only |

Efficiency tricks that recur across 2025–26 winners: **(a)** few-step diffusion via
shortcut/consistency objectives (Dreamer 4's shortcut forcing: 4 steps ≈ 64-step quality);
**(b)** SSM/linear-attention dynamics (Drama, EDELINE, SANA-WM's Gated DeltaNet);
**(c)** token frugality (Δ-IRIS delta-tokens ~10× cheaper; DDP-WM sparse "primary dynamics"
→ 9× rollout speedup); **(d)** block-causal/chunked attention for O(chunk) rollouts;
**(e)** above all, *predict latents, not pixels, over frozen encoders*. A dedicated
"quantized world model" literature barely exists yet — a visible gap.

Latent-action models matured too: Genie's trick of inferring actions from unlabeled video
now works "in the wild" (FAIR, ICML 2026) and at single-GPU scale (Jafar/Jasmine on
CoinRun), which matters when you have video but no action labels.

## 3. Three recipes that fit this machine

**Recipe A — train small from scratch (full understanding).** Dreamer/IRIS/DIAMOND-class
on toy envs and MuJoCo: minutes-to-days per run, everything inspectable. This is the
[labs ladder](../labs/README.md). Reference codebases: danijar/dreamerv3 (JAX has aarch64
CUDA-13 wheels), NM512/r2dreamer (maintained PyTorch successor), eloialonso/diamond
(12 GB documented), p-doom/jasmine (best open Genie), galilai-group/stable-worldmodel
(DINO-WM/LeWM/PLDM + 40 envs + CEM/MPPI planners; MIT; `pip install stable-worldmodel`).

**Recipe B — frozen backbone + trained head (the flagship recipe).** Replicate the
V-JEPA 2-AC / DINO-WM pattern on *your own MuJoCo robot data*: frozen DINOv2 or V-JEPA 2
encoder, train a 5M–300M action-conditioned predictor, plan with CEM/MPC. Both source
codebases are open (facebookresearch/vjepa2 ships the AC post-training code with a
single-device entry point; DINO-WM reimplemented cleanly in stable-worldmodel). Training a
~300M head over frozen features fits easily in 121 GB; wall-clock days, not weeks. This is
the highest-value compute-for-result recipe in the 2026 literature, and it produces a
*planning* robot demo, not just video.

**Recipe C — LoRA a pretrained generative WM (the fancy-video recipe).** NVIDIA's official
recipe LoRAs Cosmos-Predict2.5-2B for robot video with ~50M trainable params, 92 videos,
17 H100-hours ⇒ ≈3–4 Spark-days. Alternatives: SANA-WM 2.6B (linear DiT, LoRA support),
HY-WorldPlay-5B (training code public), AVID-style adapters over a *frozen* video model
(explicitly designed for ≤3 days × 1 A100). Output: photoreal "dreams" of your robot —
the demo-video payoff — and a WorldEval-style policy-evaluation loop.

What **not** to attempt here: full finetunes of ≥2B video models (32×A100-class),
14B/64B variants, Cosmos Policy reproduction (64×H100×48 h — though its *inference* needs
just 6.8 GB, so evaluation-side use is fine).

## 4. DGX Spark reality check

- **Throughput**: plan for ~5× slower wall-clock than H100-class and ~1.5–2.5× slower
  than an RTX 4090; ~273 GB/s LPDDR5x bandwidth is the tax. The asset is 121 GB unified
  memory — footprints that discrete 24 GB cards simply cannot hold (2B DiT + T5 + VAE
  resident together; Cosmos 2B inference at 26–65 GB).
- **FP4 is marketing here**: GB10 lacks full-rate FP4 matmul; NVFP4 is bandwidth
  compression, and FP8 measured ~32% faster than NVFP4 in practice. Train in BF16 (+FP8),
  use QLoRA for memory, ignore the "1 PFLOP" number.
- **aarch64 friction is blocker #1**: install torch from the cu130 index; **flash-attn and
  Transformer Engine don't build** (this hard-breaks the official cosmos-predict2.5
  post-training code — issue #120 — while the Cosmos 3 stack is Spark-friendly); xformers
  has no wheel; envpool is x86-only; old pins like IRIS's torch 1.11 are unfixable without
  porting. Native SDPA is actually faster on Blackwell — prefer it everywhere. JAX ships
  aarch64 CUDA-13 wheels, so the JAX codebases just work. Modern `mujoco` PyPI wheels are
  native aarch64 (old `mujoco-py` is not).
- **Unified-memory OOM freezes the whole box**: disable swap, cgroup-cap jobs (~100 GB),
  protect sshd (see natolambert/dgx-spark-setup).
- NVIDIA's own Spark playbooks include a FLUX.1-dev 12B DreamBooth LoRA — the closest
  official analog to LoRA-ing a video DiT: expect *days, not hours*.

## 5. Data generation: MuJoCo first, photorealism second

Real-robot data collection is out of scope, so data is synthetic. The engine question
resolves cleanly on this hardware:

- **MuJoCo (primary)**: native aarch64 wheels run on the Spark itself; exact, scriptable
  physics; **action labels for free** (the scarcest resource in world modeling); existing
  Unitree/Franka assembly assets in `~/work` mean unlimited (obs, action, obs′)
  trajectories today. Direct precedents for training WMs on MuJoCo manipulation data:
  DreamGen Bench (RoboCasa), Cosmos Policy (LIBERO, 98.5% from 2k demos), DINO-WM
  (PushT/rope/granular), UWM, iVideoGPT (MetaWorld).
- **Unreal Engine (not on this box)**: the UE Editor has **no Linux aarch64 support** —
  it's an open feature request for DGX Spark specifically, and cross-compiled packaged
  builds crash on GB10. Using UE means a second x86 machine exporting datasets. Isaac Sim
  is likewise x86-oriented (though Isaac playbooks exist for Spark).
- **The 2026 pattern for photorealism**: don't switch renderers — restyle. Cosmos-Transfer2.5
  does sim→photoreal transfer that preserves motion/structure (robot-multiview checkpoints,
  Isaac Sim integration; 2B inference is Spark-feasible). MuJoCo gives physics + labels;
  a pretrained transfer model gives visuals; both stay on one machine.

## 6. How this repo stays differentiated

The educational niche has one incumbent: **simchowitzlabpublic/nano-world-model** (692★,
May 2026) — nanoGPT-style minimal *diffusion-forcing video* WM. Nobody currently owns:
(a) a commit-by-commit pedagogical ladder across *all four families* on one fixed env;
(b) a tiny **latent-imagination RL** (Dreamer-style) teaching implementation — existing
Dreamer codebases are reference impls, not teaching repos; (c) **DGX-Spark/aarch64-verified**
world-model stacks — nobody documents ARM. This repo's ladder + robotics milestones +
Spark-verified setup notes cover all three gaps.

---

### Sources (per sweep)

**Frontier labs:** [V-JEPA 2](https://arxiv.org/abs/2506.09985) · [vjepa2 code](https://github.com/facebookresearch/vjepa2) · [LeCun departure](https://www.cnbc.com/2025/11/19/meta-chief-ai-scientist-yann-lecun-is-leaving-the-company-.html) · [AMI Labs $1.03B](https://techcrunch.com/2026/03/09/yann-lecuns-ami-labs-raises-1-03-billion-to-build-world-models/) · [LeJEPA](https://arxiv.org/abs/2511.08544) · [LeWM](https://le-wm.github.io/) · [Genie 3](https://deepmind.google/blog/genie-3-a-new-frontier-for-world-models/) · [Project Genie](https://blog.google/innovation-and-ai/models-and-research/google-deepmind/project-genie/) · [SIMA 2](https://deepmind.google/blog/sima-2-an-agent-that-plays-reasons-and-learns-with-you-in-virtual-3d-worlds/) · [Dreamer 4](https://arxiv.org/abs/2509.24527) · [Sora discontinuation](https://help.openai.com/en/articles/20001152-what-to-know-about-the-sora-discontinuation) · [Cosmos 3](https://huggingface.co/blog/nvidia/cosmos-3-for-physical-ai) · [SANA-WM](https://arxiv.org/abs/2605.15178) · [Marble](https://www.worldlabs.ai/blog/marble-world-model) · [RTFM](https://www.worldlabs.ai/blog/rtfm) · [HY-WorldPlay](https://github.com/Tencent-Hunyuan/HY-WorldPlay) · [HY-World 2.0](https://github.com/Tencent-Hunyuan/HY-World-2.0) · [Runway GWM-1](https://runwayml.com/research/introducing-runway-gwm-1) · [Oasis 3](https://techcrunch.com/2026/06/10/decarts-new-world-model-can-simulate-hours-of-photorealistic-driving-with-some-caveats/) · [Odyssey-2](https://odyssey.ml/introducing-odyssey-2) · [GAIA-3](https://wayve.ai/thinking/gaia-3/) · [Waymo WM](https://www.marktechpost.com/2026/02/06/waymo-introduces-the-waymo-world-model-a-new-frontier-simulator-model-for-autonomous-driving-and-built-on-top-of-genie-3/) · [1X](https://www.1x.tech/discover/redwood-ai-world-model) · [Thinking Machines](https://thinkingmachines.ai/news/)

**Small/efficient:** [DreamerV3 Nature](https://arxiv.org/abs/2301.04104) · [DIAMOND](https://arxiv.org/abs/2405.12399) · [Drama](https://arxiv.org/abs/2410.08893) · [STORM](https://arxiv.org/abs/2310.09615) · [Δ-IRIS](https://arxiv.org/abs/2406.19320) · [TWISTER](https://arxiv.org/abs/2503.04416) · [EMERALD](https://arxiv.org/abs/2507.04075) · [EDELINE](https://arxiv.org/abs/2502.00466) · [Simulus](https://arxiv.org/abs/2502.11537) · [ITC](https://arxiv.org/abs/2605.16457) · [scale probing](https://arxiv.org/abs/2605.08578) · [DINO-WM](https://arxiv.org/abs/2411.04983) · [DINO-world](https://arxiv.org/abs/2507.19468) · [PLDM](https://arxiv.org/abs/2502.14819) · [LeWM paper](https://arxiv.org/abs/2603.19312) · [Fast-LeWM](https://arxiv.org/abs/2606.26217) · [LAWM in the wild](https://arxiv.org/abs/2601.05230) · [AdaWorld](https://arxiv.org/abs/2503.18938) · [DDP-WM](https://arxiv.org/abs/2602.01780) · [efficiency survey](https://arxiv.org/abs/2603.28489) · [stable-worldmodel paper](https://arxiv.org/abs/2605.21800)

**Robotics finetuning / Spark:** [Cosmos LoRA blog](https://huggingface.co/blog/nvidia/cosmos-fine-tuning-for-robot-video-generation) · [Cosmos Cookbook](https://nvidia-cosmos.github.io/cosmos-cookbook/index.html) · [predict2.5 Spark blockers #120](https://github.com/nvidia-cosmos/cosmos-predict2.5/issues/120) · [Cosmos 3 Nano on Spark](https://dev.classmethod.jp/en/articles/dgx-spark-cosmos3-omni-world-model-policy/) · [Spark playbooks](https://github.com/nvidia/dgx-spark-playbooks) · [FLUX LoRA playbook](https://build.nvidia.com/spark/flux-finetuning) · [Spark setup gotchas](https://github.com/natolambert/dgx-spark-setup) · [Spark benchmarks vs reality](https://publish.obsidian.md/aixplore/Practical+Applications/dgx-lab-benchmarks-vs-reality-day-4) · [NVFP4 trap](https://ai-muninn.com/en/blog/dgx-spark-nvfp4-trap-gb10-fp8-wins) · [AVID](https://arxiv.org/abs/2410.12822) · [Vid2World](https://github.com/thuml/Vid2World) · [DreamGen](https://research.nvidia.com/labs/gear/dreamgen/) · [GR00T-Dreams](https://github.com/NVIDIA/GR00T-Dreams) · [Ctrl-World](https://arxiv.org/abs/2510.10125) · [WorldEval](https://arxiv.org/abs/2505.19017) · [Cosmos Policy](https://arxiv.org/abs/2601.16163) · [UWM](https://arxiv.org/abs/2504.02792) · [iVideoGPT](https://arxiv.org/abs/2405.15223) · [manipulation WM survey](https://arxiv.org/pdf/2606.00113)

**Codebases:** [dreamerv3](https://github.com/danijar/dreamerv3) · [r2dreamer](https://github.com/NM512/r2dreamer) · [diamond](https://github.com/eloialonso/diamond) · [jasmine](https://github.com/p-doom/jasmine) · [stable-worldmodel](https://github.com/galilai-group/stable-worldmodel) · [nano-world-model](https://github.com/simchowitzlabpublic/nano-world-model) · [eb_jepa](https://github.com/facebookresearch/eb_jepa) · [nicklashansen/dreamer4](https://github.com/nicklashansen/dreamer4) · [lucidrains/dreamer4](https://github.com/lucidrains/dreamer4) · [UE aarch64 feature request](https://forums.unrealengine.com/t/feature-request-official-unreal-editor-linux-arm64-aarch64-support-for-nvidia-dgx-spark-gb10/2731250) · [UE GB10 crash report](https://forums.developer.nvidia.com/t/unreal-build-for-gdx-spark-failing/361004)
