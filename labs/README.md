# Labs — the implementation ladder

Each lab is a standalone package: its own env/data, model, training loop, and a **demo
artifact** (GIF + curve) written to `media/`. Every lab adds exactly one idea over the
previous one, so diffing two adjacent labs shows you the idea in code. Every lab trains in
minutes-to-an-hour on a single GB10.

Convention: `python -m labs.labXX_name.run` trains, evaluates, and renders the demo. Each lab
directory has its own README explaining the idea, the expected result, and the demo.

| Lab | One idea added | Demo artifact |
|---|---|---|
| [00 — pixels](lab00_dynamics/) | Action-conditioned next-frame prediction; watch autoregressive rollout drift | GT vs imagination vs error GIF + PSNR-vs-horizon curve |
| 01 — latent | Train an autoencoder; predict in latent space, decode only to look | Same GIF, same budget — visibly longer coherent horizon |
| 02 — RSSM | Stochastic + deterministic state, prior/posterior, KL balancing | Sampled futures fan-out: one context, N imagined futures |
| 03 — imagination | Actor-critic trained purely inside the frozen lab-02 model | Agent solving the env; real-env return curve vs model-free baseline |
| 04 — tokens | VQ tokenizer + autoregressive transformer dynamics | Token-space rollout GIF; tokens/sec vs lab-02 |
| 05 — few-step diffusion | Diffusion dynamics head; then cut denoise steps (shortcut-style) | Fidelity vs steps grid: 1/2/4/8-step rollouts side by side |
| 06 — JEPA + MPC | Frozen pretrained encoder (DINOv2), small dynamics head, CEM planner | Goal-reaching demo: plan in embedding space, execute in env |
| 07 — evaluation | Drift, memory, physics probes + the functional test | Eval dashboard comparing labs 01–06 on one page |

Status: **lab 00 implemented**; 01–07 are specced here and built as the
[reading path](../docs/01-reading-path.md) reaches them.

## Why this ladder

It walks the four families in historical order (pixels → latent → tokens → diffusion → JEPA)
on a *fixed* environment and data budget, so every architectural claim in the literature
("tokens lose detail", "latents roll out further", "JEPA plans without a decoder") becomes a
visible A/B in your own GIFs. Labs 03 and 06 are the two "use it for control" payoffs —
imagination-RL and MPC — which are the two ways world models actually get used in robotics.

The environment is deliberately trivial (2D pushing world, exact known physics). The point of
the ladder is the *modeling* ideas; a later milestone swaps the env for a MuJoCo manipulation
scene and repeats the strongest recipe on it.
