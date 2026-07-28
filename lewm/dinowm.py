"""DINO-WM baseline (arXiv:2411.04983), small-scale variant.

The comparison the whole project builds toward: FROZEN pretrained encoder +
trained patch-grid dynamics, against LeWM's end-to-end single-token latent.
Everything downstream (CEM planner, eval protocol, demos) is shared — the
two models expose the same encode/predict/rollout interface; the state just
has different shape ((P*384,) flattened patch grid vs (192,)).

Variant notes vs the reference implementation (stable-worldmodel PreJEPA):
  * dinov2_vits14 at 98 px -> 7x7 = 49 patches (reference: 224 px, 256) —
    sized for the GB10; documented deviation.
  * action embedding (16-d) tiled and concatenated per patch token
    (reference concatenates action + proprio; we have no proprio).
  * predictor: depth 6, block-causal frame mask — frame t attends to all
    patches of frames <= t; loss on the 384-d feature slice only
    (actionless), targets DETACHED (frozen-encoder recipe: no gradient
    flows into the representation; that is the whole philosophical
    difference from LeWM)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

PATCHES = 49          # 98px / 14
FEAT = 384            # dinov2_vits14
ACT_EMB = 16
TOK = FEAT + ACT_EMB  # 400


class BlockCausalAttention(nn.Module):
    """Self-attention over (T*P) tokens where frame t attends frames <= t
    (patches within a frame attend bidirectionally)."""

    def __init__(self, dim: int, heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.heads = heads
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.out = nn.Linear(dim, dim)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, frames: int) -> torch.Tensor:
        b, n, d = x.shape
        p = n // frames
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q, k, v = (y.view(b, n, self.heads, -1).transpose(1, 2) for y in (q, k, v))
        fidx = torch.arange(n, device=x.device) // p
        mask = fidx.unsqueeze(1) >= fidx.unsqueeze(0)          # (n, n) bool
        o = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask,
            dropout_p=self.dropout if self.training else 0.0)
        return self.out(o.transpose(1, 2).reshape(b, n, d))


class Block(nn.Module):
    def __init__(self, dim: int, mlp: int = 1024, dropout: float = 0.1):
        super().__init__()
        self.n1 = nn.LayerNorm(dim)
        self.n2 = nn.LayerNorm(dim)
        self.attn = BlockCausalAttention(dim, dropout=dropout)
        self.mlp = nn.Sequential(nn.Linear(dim, mlp), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(mlp, dim),
                                 nn.Dropout(dropout))

    def forward(self, x, frames):
        x = x + self.attn(self.n1(x), frames)
        return x + self.mlp(self.n2(x))


class DinoWM(nn.Module):
    model_type = "dinowm"

    def __init__(self, action_dim: int = 10, history: int = 3,
                 depth: int = 6, dropout: float = 0.1):
        super().__init__()
        self.history = history
        self.backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
        self.backbone.eval().requires_grad_(False)
        self.act_proj = nn.Linear(action_dim, ACT_EMB)
        self.pos = nn.Parameter(torch.randn(1, history * PATCHES, TOK) * 0.02)
        self.blocks = nn.ModuleList(Block(TOK, dropout=dropout)
                                    for _ in range(depth))
        self.norm = nn.LayerNorm(TOK)
        self.head = nn.Linear(TOK, TOK)

    def train(self, mode: bool = True):
        super().train(mode)
        self.backbone.eval()                       # frozen forever
        return self

    # ------------------------------------------------- LeWM-shaped API ----

    def encode(self, obs: torch.Tensor, actions: torch.Tensor):
        """obs (B,T,3,H,W), actions (B,T,A) ->
        emb (B,T,P*384) detached features, act_emb (B,T,ACT_EMB)."""
        b, t = obs.shape[:2]
        x = obs.flatten(0, 1)
        x = F.interpolate(x, size=(98, 98), mode="bilinear", align_corners=False)
        with torch.no_grad():
            feats = self.backbone.forward_features(x)["x_norm_patchtokens"]
        emb = feats.reshape(b, t, PATCHES * FEAT).detach()
        return emb, self.act_proj(actions)

    def _predict_tokens(self, emb: torch.Tensor, act_emb: torch.Tensor):
        """(B,T,P*384),(B,T,16) -> predicted next-frame tokens (B,T,P*384)."""
        b, t = emb.shape[:2]
        tok = emb.view(b, t, PATCHES, FEAT)
        act = act_emb.unsqueeze(2).expand(-1, -1, PATCHES, -1)
        x = torch.cat([tok, act], dim=-1).view(b, t * PATCHES, TOK)
        x = x + self.pos[:, : t * PATCHES]
        for blk in self.blocks:
            x = blk(x, t)
        x = self.head(self.norm(x)).view(b, t, PATCHES, TOK)
        return x[..., :FEAT].reshape(b, t, PATCHES * FEAT)

    def predict(self, ctx_emb: torch.Tensor, ctx_act: torch.Tensor):
        return self._predict_tokens(ctx_emb, ctx_act)

    @torch.no_grad()
    def rollout(self, ctx_emb: torch.Tensor, ctx_act_emb: torch.Tensor,
                future_act_emb: torch.Tensor) -> torch.Tensor:
        embs, acts = ctx_emb, ctx_act_emb
        out = []
        for k in range(future_act_emb.size(1)):
            acts = torch.cat([acts[:, 1:], future_act_emb[:, k : k + 1]], dim=1) \
                if acts.size(1) == self.history else torch.cat(
                    [acts, future_act_emb[:, k : k + 1]], dim=1)
            window_e = embs[:, -self.history:]
            window_a = acts[:, -window_e.size(1):]
            pred = self.predict(window_e, window_a)[:, -1:]
            embs = torch.cat([embs, pred], dim=1)
            out.append(pred)
        return torch.cat(out, dim=1)

    # eval.py builds action blocks through model.action_encoder
    def action_encoder(self, a: torch.Tensor) -> torch.Tensor:
        return self.act_proj(a)


def compute_loss(model: DinoWM, obs: torch.Tensor, act: torch.Tensor):
    """Teacher-forced next-patch-feature MSE; targets DETACHED (frozen
    encoder — the anti-collapse guarantee is the freeze itself)."""
    act_padded = torch.cat([act, torch.zeros_like(act[:, :1])], dim=1)
    emb, act_emb = model.encode(obs, act_padded)
    h = model.history
    pred = model.predict(emb[:, :h], act_emb[:, :h])
    return (pred - emb[:, 1:].detach()).pow(2).mean()
