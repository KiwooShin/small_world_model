"""SIGReg — Sketched Isotropic Gaussian Regularization.        ★ TASK 1 ★

This is the anti-collapse mechanism that lets LeWM train encoder + predictor
end-to-end with NO stop-gradient, NO EMA teacher, and NO frozen backbone.
It is the single most important thing to understand in this lab.

Background (LeJEPA, arXiv:2511.08544):
    A JEPA trained only on the prediction loss ||pred(z_t, a_t) - z_{t+1}||^2
    has trivial minimizers: map every frame to the same vector (complete
    collapse) or into a low-dim subspace (dimensional collapse). LeJEPA argues
    the optimal embedding distribution — the one minimizing worst-case
    downstream risk over unknown tasks — is the isotropic Gaussian N(0, I),
    and adds one regularizer pulling the batch of embeddings toward it.
    A collapsed batch is maximally NON-Gaussian, so collapse becomes
    impossible without any architectural asymmetry.

    Testing "is this D-dim batch N(0, I)?" directly is expensive, so SIGReg
    *sketches*: project the batch onto M random unit directions (Cramér–Wold:
    if every 1-D projection is N(0,1), the joint is N(0,I)), and score each
    1-D sample with the Epps–Pulley statistic

        EP = B * Integral_t  | ecf(t) - phi(t) |^2  * w(t)  dt

    where ecf(t) = (1/B) sum_j exp(i * t * x_j) is the empirical
    characteristic function of the projected batch x (B samples),
    phi(t) = exp(-t^2 / 2) is the CF of N(0,1), and w(t) = exp(-t^2 / 2)
    is a Gaussian window confining the integral. The integral is a plain
    trapezoid quadrature over `knots` points; since the integrand is even
    in t, we integrate on [0, t_max] with doubled interior weights.

    SIGReg = the EP statistic averaged over the M random directions
    (and, in LeWM's usage, over time steps).

The __init__ below is COMPLETE — it precomputes the quadrature. Your job is
`forward`. Everything you need about tensor shapes is written there.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SIGReg(nn.Module):
    def __init__(self, knots: int = 17, num_proj: int = 1024, t_max: float = 3.0):
        super().__init__()
        self.num_proj = num_proj
        # Quadrature grid t_0..t_{P-1} uniformly on [0, t_max].
        t = torch.linspace(0.0, t_max, knots)
        dt = t_max / (knots - 1)
        # Trapezoid weights on the half-domain: interior points count twice
        # (once for +t, once for -t by symmetry of the even integrand),
        # endpoints once.
        weights = torch.full((knots,), 2.0 * dt)
        weights[0] = dt
        weights[-1] = dt
        # phi = CF of N(0,1) evaluated at the knots — this is BOTH the target
        # characteristic function and the Gaussian window w(t), so it appears
        # twice in the math and we fold one copy into the weights.
        phi = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)                    # (P,)
        self.register_buffer("phi", phi)                # (P,)
        self.register_buffer("weights", weights * phi)  # (P,)  = trapz * window

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Return the scalar SIGReg statistic for a batch of embeddings.

        z: (T, B, D) — T groups (time steps) of B embeddings of dim D.
           LeWM applies the test per time step across the batch, then
           averages: pass emb.transpose(0, 1) from a (B, T, D) tensor.

        Implement, in order:

        1. Sample a FRESH sketch matrix A of shape (D, M) with M = self.num_proj:
           i.i.d. standard-normal entries, then normalize each COLUMN to unit
           L2 norm — normalized Gaussian vectors are uniform on the sphere,
           and resampling every call is essential (a fixed set of directions
           can be gamed by the encoder; see LeJEPA Fig. 7).

        2. Project: (T, B, D) @ (D, M) -> (T, B, M). Each of the T*M columns
           is now a 1-D sample of size B to be tested against N(0, 1).

        3. Evaluate the empirical characteristic function at the knots.
           Broadcast to shape (T, B, M, P) by multiplying the projections
           (unsqueezed on a last axis) with self.t, then take cos(...) and
           sin(...) and average each over the BATCH axis, giving the real
           and imaginary parts of the ECF, each (T, M, P).
           (No complex tensors needed: |ecf - phi|^2 expands to
           (Re ecf - phi)^2 + (Im ecf)^2 because phi is real and even.)

        4. Squared CF error at each knot: (T, M, P), as in step 3's expansion.

        5. Integrate: matmul the error with self.weights -> (T, M), then
           multiply by B (the Epps–Pulley statistic scales with sample size —
           this makes the *test* calibrated, and in training it means larger
           batches push harder on Gaussianity).

        6. Return the mean over T and M — a scalar with gradients flowing
           back to z (do NOT detach anything; the whole point is that this
           term shapes the encoder).

        Sanity expectations (check.py verifies these):
          - z ~ N(0, I), B=512, D=192: statistic is small (< ~2.5)
          - z collapsed to one point: statistic is large (> ~50)
          - z Gaussian but confined to a low-dim subspace: clearly elevated.
        """
        raise NotImplementedError("TASK 1: implement SIGReg.forward")
