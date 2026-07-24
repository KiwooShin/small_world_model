"""Lab 00 model: the most literal possible world model.

f(last K frames, action) -> next frame, as one conv net trained with MSE.

No latent space, no stochasticity, no recurrence. This is the baseline the
whole ladder argues against: it works for one step and visibly falls apart
when rolled out autoregressively, because (a) MSE averages over futures and
(b) the model was never trained on its own (slightly wrong) outputs, so
errors compound — the covariate-shift problem every later lab addresses
one way or another.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ActionFiLM(nn.Module):
    """Condition conv features on the action via FiLM (scale & shift).

    FiLM is the smallest honest way to inject a continuous action into a
    conv net; concatenating action planes to the input works too but hides
    the conditioning in the first layer only.
    """

    def __init__(self, action_dim: int, channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(action_dim, 64), nn.SiLU(), nn.Linear(64, 2 * channels)
        )

    def forward(self, h: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        scale, shift = self.net(action).chunk(2, dim=-1)
        return h * (1 + scale[..., None, None]) + shift[..., None, None]


class PixelDynamics(nn.Module):
    """UNet-lite: encode K stacked frames, FiLM in the action at the
    bottleneck, decode the next frame. ~1.1M params."""

    def __init__(self, context: int = 2, action_dim: int = 2, width: int = 48):
        super().__init__()
        w = width
        self.context = context
        self.enc1 = self._block(3 * context, w)        # 64 -> 32
        self.enc2 = self._block(w, 2 * w)              # 32 -> 16
        self.enc3 = self._block(2 * w, 4 * w)          # 16 -> 8
        self.film = ActionFiLM(action_dim, 4 * w)
        self.mid = nn.Sequential(
            nn.Conv2d(4 * w, 4 * w, 3, padding=1),
            nn.GroupNorm(8, 4 * w), nn.SiLU(),
        )
        self.dec3 = self._up(4 * w, 2 * w)             # 8 -> 16
        self.dec2 = self._up(2 * w + 2 * w, w)         # 16 -> 32 (skip cat)
        self.dec1 = self._up(w + w, w)                 # 32 -> 64 (skip cat)
        self.out = nn.Conv2d(w, 3, 3, padding=1)
        # Start at the background color (sigmoid(-2.75) ~= 0.06): the constant
        # background is a strong local optimum for MSE on mostly-empty frames,
        # and without normalization + this init the net can collapse into it
        # and never recover (found the hard way — loss froze at exactly the
        # constant-predictor value 0.0129).
        nn.init.zeros_(self.out.weight)
        nn.init.constant_(self.out.bias, -2.75)

    @staticmethod
    def _block(cin: int, cout: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(cin, cout, 4, stride=2, padding=1),
            nn.GroupNorm(8, cout), nn.SiLU(),
            nn.Conv2d(cout, cout, 3, padding=1),
            nn.GroupNorm(8, cout), nn.SiLU(),
        )

    @staticmethod
    def _up(cin: int, cout: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(cin, cout, 3, padding=1),
            nn.GroupNorm(8, cout), nn.SiLU(),
        )

    def forward(self, frames: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """frames: (B, K, 3, H, W) in [0,1]; action: (B, A). Returns (B, 3, H, W).

        Predicts the *next frame directly* (not a residual): residual
        prediction is a strong trick, but this lab exists to show the naive
        thing first.
        """
        b, k, c, h, w = frames.shape
        x = frames.reshape(b, k * c, h, w)
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        m = self.mid(self.film(e3, action))
        d3 = self.dec3(m)
        d2 = self.dec2(torch.cat([d3, e2], dim=1))
        d1 = self.dec1(torch.cat([d2, e1], dim=1))
        # Sigmoid keeps rollouts in [0,1] so feeding predictions back in
        # can't blow up numerically — drift here is purely model error.
        return torch.sigmoid(self.out(d1))
