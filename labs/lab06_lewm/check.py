"""Self-checks for each TASK. Run after implementing each piece:

    python -m labs.lab06_lewm.check            # all tasks
    python -m labs.lab06_lewm.check --task 1   # just one

Unimplemented tasks are reported as SKIP, failures explain what's wrong.
Pass all four before spending GPU-hours on a real training run.
"""

from __future__ import annotations

import argparse
import sys

import torch

RESULTS: list[tuple[str, str]] = []  # (status, task)


def guarded(task: str):
    """Run a check; the check returns a detail string on success, raises
    AssertionError with an explanation on failure, and is reported SKIP
    while its target is still NotImplementedError."""
    def deco(fn):
        def run():
            try:
                detail = fn() or ""
                status = "PASS"
            except NotImplementedError:
                status, detail = "SKIP", "not implemented yet"
            except AssertionError as e:
                status, detail = "FAIL", str(e)
            except Exception as e:  # noqa: BLE001 — surface the real error to the student
                status, detail = "FAIL", f"{type(e).__name__}: {e}"
            RESULTS.append((status, task))
            print(f"[{status:^4}] {task}" + (f" — {detail}" if detail else ""))
        return run
    return deco


# ----------------------------------------------------------- TASK 1: SIGReg

@guarded("task1: SIGReg on N(0,I) is small")
def check_sigreg_gaussian():
    from .sigreg import SIGReg
    torch.manual_seed(0)
    val = SIGReg()(torch.randn(4, 512, 192)).item()
    assert 0.0 < val < 2.5, (
        f"statistic on true N(0,I) samples was {val:.3f}; expected ~0.1-2. "
        "Too large: check the batch-mean axis (must average over the B axis, "
        "size 512 here) or a missing window/weight. Negative or zero: the "
        "error term must be a sum of squares."
    )
    return f"stat={val:.3f}"


@guarded("task1: SIGReg detects collapse")
def check_sigreg_collapse():
    from .sigreg import SIGReg
    torch.manual_seed(0)
    s = SIGReg()
    point = torch.randn(1, 1, 192).expand(1, 512, 192).contiguous()
    v_point = s(point).item()
    assert v_point > 50, (
        f"complete collapse (all 512 embeddings identical) scored {v_point:.2f}; "
        "expected a large statistic (>50). Did you forget to scale by the "
        "batch size B (step 5)?"
    )
    lowrank = torch.randn(1, 512, 8) @ torch.randn(1, 8, 192) / (8 ** 0.5)
    v_low = s(lowrank).item()
    v_gauss = s(torch.randn(1, 512, 192)).item()
    assert v_low > 4 * v_gauss, (
        f"rank-8 embeddings scored {v_low:.2f} vs {v_gauss:.2f} for full-rank "
        "Gaussian — dimensional collapse should be clearly elevated. Are your "
        "projection directions unit-norm columns (normalize dim=0 of a (D,M) "
        "matrix)?"
    )
    return f"point={v_point:.1f} rank8={v_low:.2f} gauss={v_gauss:.2f}"


@guarded("task1: SIGReg is differentiable")
def check_sigreg_grad():
    from .sigreg import SIGReg
    z = torch.randn(2, 128, 192, requires_grad=True)
    SIGReg(num_proj=64)(z).backward()
    assert z.grad is not None and z.grad.abs().sum() > 0, (
        "no gradient reached the input — do not wrap the statistic in "
        "torch.no_grad() and do not detach the projections"
    )


# ------------------------------------------------- TASK 2: ConditionalBlock

@guarded("task2: AdaLN-zero block is identity at init")
def check_block_identity():
    from .model import ConditionalBlock
    torch.manual_seed(0)
    blk = ConditionalBlock(192).eval()
    x = torch.randn(4, 3, 192)
    out = blk(x, torch.randn(4, 3, 192))
    assert torch.allclose(out, x, atol=1e-6), (
        f"block at init must return its input exactly (max dev "
        f"{(out - x).abs().max():.2e}). Both residual branches must be "
        "multiplied by a gate that is zero at init — zero shift/scale alone "
        "is not enough."
    )


def _nudge_adaln(blk) -> bool:
    """Simulate one optimizer step on the AdaLN head."""
    found = False
    for m in blk.modules():
        if isinstance(m, torch.nn.Linear) and m.out_features == 6 * 192:
            torch.nn.init.normal_(m.weight, std=0.02)
            torch.nn.init.normal_(m.bias, std=0.02)
            found = True
    return found


@guarded("task2: actions modulate after training starts")
def check_block_conditioning():
    from .model import ConditionalBlock
    torch.manual_seed(0)
    blk = ConditionalBlock(192).eval()
    assert _nudge_adaln(blk), (
        "no Linear with out_features == 6*dim found — the AdaLN head must "
        "produce 6 chunks (shift/scale/gate for attn and mlp)"
    )
    x = torch.randn(2, 3, 192)
    o1 = blk(x, torch.randn(2, 3, 192))
    o2 = blk(x, torch.randn(2, 3, 192))
    assert not torch.allclose(o1, o2, atol=1e-5), (
        "different action embeddings produced identical outputs — the c "
        "argument is not reaching the modulation"
    )


@guarded("task2: block is causal")
def check_block_causal():
    from .model import ConditionalBlock
    torch.manual_seed(0)
    blk = ConditionalBlock(192).eval()
    _nudge_adaln(blk)
    x = torch.randn(1, 3, 192)
    c = torch.randn(1, 3, 192)
    o1 = blk(x, c)
    x2 = x.clone()
    x2[:, -1] += 10.0
    o2 = blk(x2, c)
    assert torch.allclose(o1[:, :-1], o2[:, :-1], atol=1e-5), (
        "perturbing the LAST frame changed EARLIER outputs — attention must "
        "be causal (is_causal=True is already set in CausalAttention; did you "
        "bypass it?)"
    )


# ----------------------------------------------------- TASK 3: compute_loss

@guarded("task3: loss composition and shapes")
def check_loss_composition():
    from .model import LeWM
    from .sigreg import SIGReg
    from .train import compute_loss
    torch.manual_seed(0)
    model = LeWM().eval()
    obs = torch.rand(8, 4, 3, 64, 64)
    act = torch.rand(8, 3, 2) * 2 - 1
    loss, pred_l, sig_l = compute_loss(model, SIGReg(num_proj=64), obs, act, 0.09)
    for name, v in (("loss", loss), ("pred_loss", pred_l), ("sigreg_loss", sig_l)):
        assert torch.is_tensor(v) and v.dim() == 0, f"{name} must be a scalar tensor"
    assert torch.allclose(loss, pred_l + 0.09 * sig_l, rtol=1e-4), (
        "loss != pred + lambda * sigreg — check the weighting"
    )


@guarded("task3: no stop-gradient on the target path")
def check_loss_end_to_end():
    from .model import LeWM
    from .sigreg import SIGReg
    from .train import compute_loss
    torch.manual_seed(0)
    model = LeWM().eval()
    # Cut the prediction path: predictions become constants, so any encoder
    # gradient must arrive through the TARGET embeddings. lambda=0 removes
    # the SIGReg path too.
    model.predict = lambda ctx_emb, ctx_act: torch.zeros_like(ctx_emb).detach()
    obs = torch.rand(4, 4, 3, 64, 64)
    act = torch.rand(4, 3, 2)
    loss, _, _ = compute_loss(model, SIGReg(num_proj=64), obs, act, 0.0)
    loss.backward()
    g = sum(p.grad.abs().sum() for p in model.encoder.parameters()
            if p.grad is not None)
    assert g > 0, (
        "with the prediction path cut, NO gradient reached the encoder — "
        "you detached the targets. LeWM's whole point is that targets are "
        "NOT detached (no stop-gradient, no EMA); SIGReg is what makes this "
        "safe. Remove the .detach()."
    )


# ------------------------------------------------------------- TASK 4: CEM

class _PointMass:
    """Fake LeWM: latent dims 0-1 are a 2-D position read off the images'
    channel means; actions move the point by 0.1 per step. Lets the CEM
    check run without a trained model."""

    history = 3

    def __init__(self):
        self.action_encoder = lambda a: torch.nn.functional.pad(a, (0, 190))

    def encode(self, obs, actions):
        pos = obs.float().mean(dim=(3, 4))[..., :2] * 10  # (B, T, 2)
        z = torch.nn.functional.pad(pos, (0, 190))
        return z, self.action_encoder(actions)

    def rollout(self, ctx_emb, ctx_act_emb, future_act_emb):
        z = ctx_emb[:, -1]
        out = []
        for k in range(future_act_emb.size(1)):
            z = z + 0.1 * torch.nn.functional.pad(
                future_act_emb[:, k, :2], (0, 190))
            out.append(z.unsqueeze(1))
        return torch.cat(out, dim=1)


@guarded("task4: CEM reaches a reachable goal")
def check_cem():
    from .planner import CEMConfig, CEMPlanner
    torch.manual_seed(0)
    planner = CEMPlanner(_PointMass(),
                         CEMConfig(horizon=8, samples=128, elites=16, iters=8))
    ctx = torch.full((3, 3, 64, 64), 0.02)          # start position ~(0.2, 0.2)
    goal = torch.full((3, 64, 64), 0.06)            # goal position ~(0.6, 0.6)
    plan = planner.plan(ctx, torch.zeros(3, 2), goal)
    assert tuple(plan.shape) == (8, 2), \
        f"plan shape {tuple(plan.shape)}, expected (8, 2)"
    assert plan.abs().max() <= 1.0 + 1e-5, "actions must be clamped to [-1, 1]"
    start = torch.full((2,), 0.2)
    goal_pos = torch.full((2,), 0.6)
    end = start + 0.1 * plan.sum(dim=0).cpu()
    dist = (end - goal_pos).norm().item()
    assert dist < 1.0, (
        f"executing the plan lands {dist:.2f} latent units from the goal "
        "(start-goal distance was ~5.7). The CEM loop is not optimizing: "
        "check that elites are the LOWEST-cost samples and that mean/std are "
        "refit from them each iteration."
    )
    return f"final dist {dist:.2f}"


# -------------------------------------------------------------------- main

CHECKS = {
    1: [check_sigreg_gaussian, check_sigreg_collapse, check_sigreg_grad],
    2: [check_block_identity, check_block_conditioning, check_block_causal],
    3: [check_loss_composition, check_loss_end_to_end],
    4: [check_cem],
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", type=int, choices=sorted(CHECKS), default=None)
    args = ap.parse_args()
    for t in ([args.task] if args.task else sorted(CHECKS)):
        print(f"--- task {t} ---")
        for fn in CHECKS[t]:
            fn()
    fails = sum(s == "FAIL" for s, _ in RESULTS)
    skips = sum(s == "SKIP" for s, _ in RESULTS)
    print(f"\n{len(RESULTS) - fails - skips} passed, {fails} failed, {skips} skipped")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
