# Roadmap

Goal of the project: demonstrate, with working code and demo videos, four things —
**(1)** studied understanding of world models, **(2)** ability to build and train them,
**(3)** application to robotics, **(4)** serious treatment of efficiency (everything runs
on one DGX Spark). Every milestone ends in a demo artifact in the README.

*Restructured 2026-07-28: the incremental labs ladder was retired (git history has it);
the project now centers on one model — LeWM — taken from a simple MuJoCo task toward
contact-rich manipulation.* Recipes and feasibility numbers behind these choices:
[02-small-wm-frontier-2026.md](02-small-wm-frontier-2026.md).

## M0 — Study foundation ✅ (2026-07-24)

Landscape map, reading path, research sweep (frontier labs, small/efficient WMs, DGX
Spark constraints), notes system. LeWM implementation components (SIGReg, AdaLN-zero
predictor, end-to-end loss, CEM) written and verified against the papers and both public
codebases.

## M1 — LeWM on a simple MuJoCo task 🔨 (active)

From-scratch LeWM ([lewm/](../lewm/)) trained on a 2-DoF MuJoCo reacher from pixels:
offline episodes from a scripted policy, per-epoch collapse dashboard (pred / sigreg /
lat_std / eff_rank), CEM goal-image planning eval with success-rate scores, and the λ=0
collapse ablation.

- **Demo:** training curves; goal-image planning GIF (goal | imagination | execution);
  success-rate table incl. the collapsed model as the control.

## M2 — Scale the task ladder

Same model, harder MuJoCo scenes: planar pushing (first contact dynamics), then the
existing Franka/Unitree assembly assets as the data engine. Add the paper's frame-skip-5
action blocks, tune data volume/epochs per task. This is where "does one 192-dim token
survive contact?" starts getting an answer.

- **Demo:** per-task success table; side-by-side plans on easy vs contact-rich tasks.

## M3 — LeWM vs DINO-WM, head to head

The open question nobody has tested: end-to-end 192-dim LeWM vs frozen-DINOv2 patch-grid
DINO-WM on *contact-rich manipulation*, same data, same CEM protocol
(stable-worldmodel as the DINO-WM reference). Either outcome is a finding.

- **Demo:** head-to-head success/precision table; where each fails, shown.

## M4 — V-JEPA 2-AC-style post-training

Frozen video encoder + block-causal action-conditioned predictor (~100–300M) on our
MuJoCo data; zero-shot planning in held-out scenes. The strongest compute-for-result
recipe of 2026, replicated.

## M5 — Efficiency + generative story

LoRA Cosmos-Predict2.5-2B or SANA-WM on our robot videos → photoreal robot dreams;
honest efficiency table (params, wall-clock, memory, FPS) across everything built.

---

### Standing constraints (locked)

- Single DGX Spark; BF16 (+FP8) training; SDPA everywhere; no flash-attn/TE/xformers.
- MuJoCo for data (native aarch64). Unreal Engine only if a second x86 machine appears;
  photorealism via transfer models otherwise.
- Every milestone: README artifact + a short "what this shows" caption. Small models,
  full understanding — no cluster jobs.
- Push to remote after every meaningful change.
