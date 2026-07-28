"""SIGReg — Sketched Isotropic Gaussian Regularization (LeJEPA, arXiv:2511.08544).

The anti-collapse mechanism that lets LeWM train encoder + predictor end-to-end
with no stop-gradient, no EMA teacher, and no frozen backbone.

A JEPA trained only on ||pred(z_t, a_t) - z_{t+1}||^2 has trivial minimizers:
map every frame to the same vector (complete collapse) or into a low-dim
subspace (dimensional collapse). LeJEPA's answer: the optimal embedding
distribution under an unknown downstream task is the isotropic Gaussian
N(0, I), so add one term pulling the batch of embeddings toward it. A
collapsed batch is maximally non-Gaussian, so the degenerate minima stop
being minima of the total loss.

Testing "is this D-dim batch N(0, I)?" directly is expensive, so SIGReg
*sketches* (Cramér–Wold: if every 1-D projection is N(0,1), the joint is
N(0, I)): project onto M random unit directions, resampled fresh every call
so the encoder can't game a fixed set, and score each 1-D sample with the
Epps–Pulley statistic

    EP = B * Integral_t  | ecf(t) - phi(t) |^2 * w(t)  dt

where ecf is the empirical characteristic function of the projected batch,
phi(t) = exp(-t^2/2) is the CF of N(0,1), and w(t) = exp(-t^2/2) a Gaussian
window. The integral is a trapezoid quadrature on [0, t_max] with doubled
interior weights (the integrand is even in t).

Calibration on D=192 (B=512): true N(0,I) scores ~1, complete collapse ~500,
rank-8 embeddings ~16.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SIGReg(nn.Module):
    def __init__(self, knots: int = 17, num_proj: int = 1024, t_max: float = 3.0):
        super().__init__()
        self.num_proj = num_proj
        t = torch.linspace(0.0, t_max, knots)
        dt = t_max / (knots - 1)
        weights = torch.full((knots,), 2.0 * dt)  # interior knots count twice (+t and -t)
        weights[0] = dt
        weights[-1] = dt
        phi = torch.exp(-t.square() / 2.0)  # CF of N(0,1) == the window, so it
        self.register_buffer("t", t)        # appears squared overall: once here,
        self.register_buffer("phi", phi)    # once folded into the weights.
        self.register_buffer("weights", weights * phi)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: (T, B, D) — T groups of B embeddings, tested independently
        against N(0, I) across the batch axis and averaged. LeWM applies this
        per time step: pass emb.transpose(0, 1) from a (B, T, D) tensor."""
        A = torch.randn(z.size(-1), self.num_proj, device=z.device, dtype=z.dtype)
        A = A / A.norm(p=2, dim=0)                      # columns uniform on sphere
        proj = z @ A                                    # (T, B, M)
        x_t = proj.unsqueeze(-1) * self.t               # (T, B, M, P)
        ecf_re = x_t.cos().mean(dim=-3)                 # (T, M, P) empirical CF
        ecf_im = x_t.sin().mean(dim=-3)                 # (phi is real+even, so
        err = (ecf_re - self.phi).square() + ecf_im.square()  # |ecf-phi|^2 splits)
        stat = (err @ self.weights) * z.size(-2)        # (T, M), scaled by B
        return stat.mean()
