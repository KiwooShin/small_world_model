"""CEM planning in latent space — where the world model pays rent.

Given a goal IMAGE, find the action sequence whose imagined outcome matches
it: encode the goal once, roll candidate action sequences through the
predictor, score by MSE to the goal latent at the last step, refit a
Gaussian to the elites. No reward function, no policy network, no decoder.

Official benchmark values (stable-worldmodel `cem.yaml`): 300 samples,
30 elites, 30 iterations. The lab default is lighter; raise `iters` first
if success rates disappoint.
"""

from __future__ import annotations

import dataclasses

import torch

from .model import LeWM


@dataclasses.dataclass
class CEMConfig:
    horizon: int = 8      # actions to plan
    samples: int = 256    # candidates per iteration
    elites: int = 32      # top-k kept for refitting
    iters: int = 10       # refinement rounds
    var: float = 1.0      # initial sampling std (actions live in [-1, 1])


class CEMPlanner:
    def __init__(self, model: LeWM, cfg: CEMConfig = CEMConfig()):
        self.model = model
        self.cfg = cfg

    @torch.no_grad()
    def plan(self, ctx_frames: torch.Tensor, ctx_actions: torch.Tensor,
             goal_frame: torch.Tensor) -> torch.Tensor:
        """ctx_frames (H, 3, h, w), ctx_actions (H, A) — last row is a zero
        placeholder (the action at the newest frame is what we're choosing);
        goal_frame (3, h, w). Returns a (horizon, A) action plan."""
        cfg, model = self.cfg, self.model
        dev = ctx_frames.device
        z_goal, _ = model.encode(
            goal_frame.unsqueeze(0).unsqueeze(0),
            torch.zeros(1, 1, ctx_actions.size(-1), device=dev))
        z_goal = z_goal[:, 0]                                    # (1, D)
        ctx_emb, ctx_act_emb = model.encode(
            ctx_frames.unsqueeze(0), ctx_actions.unsqueeze(0))
        ctx_emb = ctx_emb.expand(cfg.samples, -1, -1)
        ctx_act_emb = ctx_act_emb.expand(cfg.samples, -1, -1)

        a_dim = ctx_actions.size(-1)
        mean = torch.zeros(cfg.horizon, a_dim, device=dev)
        std = torch.full((cfg.horizon, a_dim), cfg.var, device=dev)
        for _ in range(cfg.iters):
            cand = mean + std * torch.randn(
                cfg.samples, cfg.horizon, a_dim, device=dev)
            cand[0] = mean                       # incumbent always evaluated
            cand = cand.clamp(-1, 1)
            fut = model.action_encoder(cand)
            z_end = model.rollout(ctx_emb, ctx_act_emb, fut)[:, -1]
            cost = (z_end - z_goal).pow(2).sum(dim=-1)           # GoalMSE
            elite = cand[cost.topk(cfg.elites, largest=False).indices]
            mean, std = elite.mean(dim=0), elite.std(dim=0) + 1e-6
        return mean.clamp(-1, 1)                 # official: return refit mean
