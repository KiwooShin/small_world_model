"""Goal-image planning evaluation with scores.

    python -m lewm.eval [--ckpt data/ckpt/reacher.pt] [--episodes 25]

Protocol (the stable-worldmodel benchmark trick, adapted):
  1. Reset the env; record `history` context frames.
  2. Snapshot the sim state; roll the scripted policy N steps; the resulting
     frame is the GOAL IMAGE (and its fingertip position the true goal);
     teleport back to the snapshot.
  3. MPC loop: CEM-plan toward the goal image, execute the first
     `replan_every` actions, replan; stop when the true fingertip is within
     `success_dist` of the goal fingertip or the budget runs out.
Scores: success rate + mean/median final fingertip distance (meters).
The arm's full reach is 0.23 m; random final poses land ~0.2 m away on
average, so distances well below 0.1 mean real planning.

Also writes media/lewm_<tag>_plan.gif: [goal | live execution] side by side
for the first few episodes.
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
             replan_every: int = 4, seed: int = 42, device: str = "cuda",
             gif_tag: str | None = None, baseline: str | None = None,
             cem: "CEMConfig | None" = None, warm: bool = False) -> dict:
    """baseline: None (plan with the model), 'zero' (no-op policy), or
    'random' (uniform actions). The baselines anchor every score against
    chance — if they score high, the task is broken, not the model."""
    planner = CEMPlanner(model, cem or CEMConfig())
    env = make(env_name, seed=seed)
    budget, success_dist = env.EVAL_BUDGET, env.SUCCESS_DIST
    rng = np.random.default_rng(seed)
    t = lambda x: torch.as_tensor(np.array(x), dtype=torch.float32, device=device)

    dists, start_dists, wins, gif_rows = [], [], 0, []
    for ep in range(episodes):
        env.reset()
        frames = [env.render()]
        acts = []
        for _ in range(model.history - 1):
            a = rng.uniform(-0.5, 0.5, size=env.action_dim).astype(np.float32)
            frames.append(env.step(a))
            acts.append(a)
        acts.append(np.zeros(env.action_dim, dtype=np.float32))  # placeholder

        _, goal_img, goal_tip = env.sample_goal()   # pose-space goal
        start_dists.append(float(np.linalg.norm(env.target_point - goal_tip)))

        ep_frames = []
        done = False
        prev_plan = None
        for _ in range(0, budget, replan_every):
            if baseline == "zero":
                plan = np.zeros((replan_every, env.action_dim), dtype=np.float32)
            elif baseline == "random":
                plan = rng.uniform(-1, 1, (replan_every, env.action_dim)).astype(np.float32)
            else:
                ctx_f = t(frames[-model.history:]).permute(0, 3, 1, 2)
                ctx_a = t(np.stack(acts[-model.history:]))
                wm = None
                if warm and prev_plan is not None:
                    wm = torch.cat([prev_plan[replan_every:],
                                    torch.zeros(replan_every, prev_plan.size(-1),
                                                device=prev_plan.device)])[:planner.cfg.horizon]
                plan_t = planner.plan(ctx_f, ctx_a,
                                      t(goal_img).permute(2, 0, 1), warm_mean=wm)
                prev_plan = plan_t
                plan = plan_t.cpu().numpy()
            for a in plan[:replan_every]:
                frames.append(env.step(a))
                ep_frames.append(frames[-1])
                acts[-1] = a.astype(np.float32)
                acts.append(np.zeros(env.action_dim, dtype=np.float32))
                if np.linalg.norm(env.target_point - goal_tip) < success_dist:
                    done = True            # first-passage success
                    break
            if done:
                break
        d = float(np.linalg.norm(env.target_point - goal_tip))
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
    ap.add_argument("--gif", action="store_true", default=True)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    blob = torch.load(args.ckpt, map_location=dev)
    model = LeWM(history=blob.get("history_len", 3),
                 action_dim=blob.get("action_dim", 2)).to(dev).eval()
    model.load_state_dict(blob["model"])
    tag = args.ckpt.stem if (args.gif and not args.baseline) else None
    env_name = args.env or blob.get("env", "reacher")
    cem = CEMConfig(horizon=args.horizon, iters=args.iters,
                    cost_steps=args.cost_steps)
    evaluate(model, env_name=env_name, episodes=args.episodes, device=dev,
             gif_tag=tag, baseline=args.baseline, cem=cem, warm=args.warm)


if __name__ == "__main__":
    main()
