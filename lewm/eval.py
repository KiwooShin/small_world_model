"""Goal-image planning evaluation with scores.

    python -m lewm.eval [--ckpt data/ckpt/reacher.pt] [--episodes 25]

Protocol: reset; build `history` context frames (spaced `frameskip` env
steps apart, matching training); sample a pose-space goal; MPC with CEM in
latent space; success = first passage of the env's target point within
SUCCESS_DIST of the goal point. Scores anchored by --baseline zero/random
rows — if a baseline scores high, the task is broken, not the model.

With frameskip fs > 1 the planner thinks in action BLOCKS (fs raw actions);
executing one block = fs env steps. This is the official LeWM setting
(fs=5) and is load-bearing: SIGReg-whitened latent distance is only
informative within a short radius, so goals must be few MODEL steps away.
"""

from __future__ import annotations

import argparse
import pathlib

import numpy as np
import torch

from .envs import make
from .model import LeWM
from .planner import CEMConfig, CEMPlanner

MEDIA = pathlib.Path("media")


@torch.no_grad()
def evaluate(model: LeWM, env_name: str = "reacher", episodes: int = 25,
             frameskip: int = 1, replan_every: int | None = None,
             seed: int = 42, device: str = "cuda",
             gif_tag: str | None = None, baseline: str | None = None,
             cem: CEMConfig | None = None, warm: bool = False) -> dict:
    planner = CEMPlanner(model, cem or CEMConfig())
    env = make(env_name, seed=seed)
    fs = frameskip
    budget_blocks = max(1, env.EVAL_BUDGET // fs)
    success_dist = env.SUCCESS_DIST
    if replan_every is None:
        replan_every = 4 if fs == 1 else 1          # blocks between replans
    rng = np.random.default_rng(seed)
    t = lambda x: torch.as_tensor(np.array(x), dtype=torch.float32, device=device)
    a_raw = env.action_dim

    dists, start_dists, wins, gif_rows = [], [], 0, []
    for ep in range(episodes):
        env.reset()
        frames = [env.render()]                     # frames at block boundaries
        blocks = []                                 # executed action blocks
        for _ in range(model.history - 1):
            blk = rng.uniform(-0.5, 0.5, size=(fs, a_raw)).astype(np.float32)
            for a in blk:
                f = env.step(a)
            frames.append(f)
            blocks.append(blk.reshape(-1))
        blocks.append(np.zeros(fs * a_raw, dtype=np.float32))  # placeholder

        for _ in range(4):                     # guard: goal must be nontrivial
            _, goal_img, goal_pt = env.sample_goal()
            if np.linalg.norm(env.target_point - goal_pt) >= success_dist + 0.015:
                break
        start_dists.append(float(np.linalg.norm(env.target_point - goal_pt)))

        ep_frames = []
        done = False
        prev_plan = None
        executed = 0
        while executed < budget_blocks and not done:
            n_exec = min(replan_every, budget_blocks - executed)
            if baseline == "zero":
                plan = np.zeros((n_exec, fs * a_raw), dtype=np.float32)
            elif baseline == "random":
                plan = rng.uniform(-1, 1, (n_exec, fs * a_raw)).astype(np.float32)
            else:
                ctx_f = t(frames[-model.history:]).permute(0, 3, 1, 2)
                ctx_a = t(np.stack(blocks[-model.history:]))
                wm = None
                if warm and prev_plan is not None:
                    wm = torch.cat([prev_plan[n_exec:],
                                    torch.zeros(n_exec, prev_plan.size(-1),
                                                device=prev_plan.device)])[:planner.cfg.horizon]
                plan_t = planner.plan(ctx_f, ctx_a,
                                      t(goal_img).permute(2, 0, 1), warm_mean=wm)
                prev_plan = plan_t
                plan = plan_t.cpu().numpy()
            for blk in plan[:n_exec]:
                for a in blk.reshape(fs, a_raw):
                    f = env.step(a)
                    ep_frames.append(f)
                    if np.linalg.norm(env.target_point - goal_pt) < success_dist:
                        done = True                 # first-passage success
                        break
                frames.append(f)
                blocks[-1] = blk.astype(np.float32)
                blocks.append(np.zeros(fs * a_raw, dtype=np.float32))
                executed += 1
                if done:
                    break
        d = float(np.linalg.norm(env.target_point - goal_pt))
        dists.append(d)
        wins += done
        print(f"  ep {ep+1:2d}: {'success' if done else 'fail   '}  "
              f"start {start_dists[-1]:.3f} -> final {d:.3f} m")
        if gif_tag and ep < 4:
            gif_rows.append((goal_img, ep_frames))

    result = {"success_rate": wins / episodes,
              "mean_dist": float(np.mean(dists)),
              "median_dist": float(np.median(dists)),
              "mean_start_dist": float(np.mean(start_dists))}
    print(f"\n[{baseline or 'model'}] success rate: "
          f"{result['success_rate']:.0%} ({wins}/{episodes})   "
          f"mean dist {result['mean_dist']:.3f} m (start {result['mean_start_dist']:.3f})   "
          f"median {result['median_dist']:.3f} m   "
          f"(success threshold {success_dist} m)")
    if gif_tag and gif_rows:
        _write_gif(gif_rows, gif_tag)
    return result


def _write_gif(rows, tag: str) -> None:
    import imageio.v2 as imageio

    MEDIA.mkdir(exist_ok=True)
    n = max(len(fr) for _, fr in rows)
    frames_out = []
    for i in range(n):
        panels = []
        for goal, fr in rows:
            live = fr[min(i, len(fr) - 1)]
            sep = np.ones((64, 2, 3), dtype=np.float32)
            panels.append(np.concatenate([goal, sep, live], axis=1))
        grid = np.concatenate(panels, axis=0)
        grid = np.kron(grid, np.ones((3, 3, 1))).clip(0, 1)  # upscale 3x
        frames_out.append((grid * 255).astype(np.uint8))
    out = MEDIA / f"lewm_{tag}_plan.gif"
    imageio.mimsave(out, frames_out, fps=8, loop=0)
    print(f"wrote {out}  (left: goal image, right: MPC execution)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=pathlib.Path, default="data/ckpt/reacher.pt")
    ap.add_argument("--env", type=str, default=None,
                    help="env name; defaults to the one stored in the ckpt")
    ap.add_argument("--episodes", type=int, default=25)
    ap.add_argument("--baseline", choices=["zero", "random"], default=None)
    ap.add_argument("--horizon", type=int, default=8)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--cost-steps", type=int, default=1)
    ap.add_argument("--warm", action="store_true")
    ap.add_argument("--replan", type=int, default=None)
    ap.add_argument("--gif", action="store_true", default=True)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    blob = torch.load(args.ckpt, map_location=dev)
    if blob.get("model_type") == "dinowm":
        from .dinowm import DinoWM
        model = DinoWM(action_dim=blob.get("action_dim", 10),
                       history=blob.get("history_len", 3)).to(dev).eval()
    else:
        model = LeWM(history=blob.get("history_len", 3),
                     action_dim=blob.get("action_dim", 2)).to(dev).eval()
    model.load_state_dict(blob["model"])
    tag = args.ckpt.stem if (args.gif and not args.baseline) else None
    env_name = args.env or blob.get("env", "reacher")
    cem = CEMConfig(horizon=args.horizon, iters=args.iters,
                    cost_steps=args.cost_steps)
    evaluate(model, env_name=env_name, episodes=args.episodes, device=dev,
             frameskip=blob.get("frameskip", 1), replan_every=args.replan,
             gif_tag=tag, baseline=args.baseline, cem=cem, warm=args.warm)


if __name__ == "__main__":
    main()
