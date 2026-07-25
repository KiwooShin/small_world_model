"""Planning with the world model: CEM in latent space.          ★ TASK 4 ★

This is where the world model pays rent: given a GOAL IMAGE, find actions
whose imagined outcome matches it — no reward function, no policy network,
just the frozen world model and a search.

    python -m labs.lab06_lewm.planner --ckpt data/lab06_ckpt/lewm.pt

The eval harness (provided, bottom) replays the dataset trick used by the
LeWM/stable-worldmodel benchmark: teleport the env to a start state, take
the goal to be where a scripted policy ends up 15 steps later, give the
planner the goal *image*, and count success by true ball position — the
planner never sees state, only pixels.
"""

from __future__ import annotations

import argparse
import dataclasses
import pathlib

import numpy as np
import torch

from labs.lab00_dynamics.env import PushWorld
from .model import LeWM


@dataclasses.dataclass
class CEMConfig:
    """Official benchmark values: samples 300, elites 30, iters 30.
    Lab defaults are lighter; crank them if success rates disappoint."""
    horizon: int = 8          # actions to plan (official: 5 blocks of 5)
    samples: int = 256        # candidates per iteration
    elites: int = 32          # top-k kept for refitting
    iters: int = 10           # CEM refinement rounds
    var: float = 1.0          # initial sampling std (actions live in [-1,1])


class CEMPlanner:
    def __init__(self, model: LeWM, cfg: CEMConfig = CEMConfig()):
        self.model = model
        self.cfg = cfg

    @torch.no_grad()
    def plan(self, ctx_frames: torch.Tensor, ctx_actions: torch.Tensor,
             goal_frame: torch.Tensor) -> torch.Tensor:
        """Return a (horizon, A) action plan.

        ctx_frames  (H, 3, 64, 64)  the last H observed frames
        ctx_actions (H, A)          actions taken at those frames (the last
                                    entry is a zero placeholder — the action
                                    at the newest frame is what we're choosing)
        goal_frame  (3, 64, 64)     the goal image

        The Cross-Entropy Method, as in the official benchmark:

        1. Encode the goal frame and the context frames ONCE (batch dim 1;
           model.encode wants (B, T, ...)). The goal's latent z_goal is the
           target. Context latents/action-embs get expanded to the sample
           batch (`.expand` is fine — rollout doesn't write in place).

        2. Maintain a Gaussian over action sequences: mean (horizon, A)
           initialized to zeros, std initialized to cfg.var.

        3. Repeat cfg.iters times:
             a. Sample cfg.samples sequences ~ N(mean, std), clamp to
                [-1, 1] (the env clips anyway; clamping keeps the search
                honest). Official detail worth copying: overwrite sample 0
                with the current mean, so the incumbent is always evaluated.
             b. Embed the candidate actions with model.action_encoder, roll
                out with model.rollout(...), take the LAST predicted latent.
             c. Score: MSE to z_goal per candidate (this is the benchmark's
                GoalMSE objective — last step only, summed over dims).
             d. Refit mean and std to the cfg.elites lowest-cost sequences.

        4. Return the final mean (not the best sample — the official solver
           returns the refit mean; it's smoother and just as good).

        Speed note: all cfg.samples candidates should go through rollout as
        ONE batch, not a Python loop over candidates.
        """
        raise NotImplementedError("TASK 4: implement CEMPlanner.plan")


# ------------------------------------------------------ provided eval ------

def _get_state(env: PushWorld):
    return (env.pusher.copy(), env.ball.copy(), env.ball_vel.copy())


def _set_state(env: PushWorld, s) -> None:
    env.pusher, env.ball, env.ball_vel = (x.copy() for x in s)


@torch.no_grad()
def evaluate(model: LeWM, episodes: int = 20, budget: int = 30,
             replan_every: int = 4, success_dist: float = 0.10,
             seed: int = 42, device: str = "cuda") -> float:
    """MPC evaluation. Success = true ball position within `success_dist`
    of the goal ball position within `budget` steps. Returns success rate."""
    cfg = CEMConfig()
    planner = CEMPlanner(model, cfg)
    env = PushWorld(seed=seed)
    rng = np.random.default_rng(seed)
    wins = 0
    for ep in range(episodes):
        env.reset()
        # Build H context frames by taking H-1 small random steps.
        frames = [env.render()]
        acts = []
        for _ in range(model.history - 1):
            a = rng.uniform(-0.3, 0.3, size=2).astype(np.float32)
            frames.append(env.step(a))
            acts.append(a)
        acts.append(np.zeros(2, dtype=np.float32))  # placeholder at newest frame
        start = _get_state(env)

        # Goal: where the scripted policy ends up 15 steps from here.
        for _ in range(15):
            env.step(env.scripted_action())
        goal_img, goal_ball = env.render(), env.ball.copy()
        _set_state(env, start)

        t = lambda x: torch.as_tensor(np.array(x), dtype=torch.float32, device=device)
        for step in range(0, budget, replan_every):
            ctx_f = t(frames[-model.history:]).permute(0, 3, 1, 2)
            ctx_a = t(acts[-model.history:])
            plan = planner.plan(ctx_f, ctx_a, t(goal_img).permute(2, 0, 1))
            for a in plan[:replan_every].cpu().numpy():
                frames.append(env.step(a))
                acts[-1] = a.astype(np.float32)      # fill the placeholder
                acts.append(np.zeros(2, dtype=np.float32))
            if np.linalg.norm(env.ball - goal_ball) < success_dist:
                wins += 1
                break
        print(f"  ep {ep+1:2d}: {'success' if np.linalg.norm(env.ball - goal_ball) < success_dist else 'fail'}"
              f"  (ball-goal dist {np.linalg.norm(env.ball - goal_ball):.3f})")
    rate = wins / episodes
    print(f"success rate: {rate:.0%}  ({wins}/{episodes})")
    return rate


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=pathlib.Path, default="data/lab06_ckpt/lewm.pt")
    ap.add_argument("--episodes", type=int, default=20)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = LeWM().to(dev).eval()
    model.load_state_dict(torch.load(args.ckpt, map_location=dev)["model"])
    evaluate(model, episodes=args.episodes, device=dev)


if __name__ == "__main__":
    main()
