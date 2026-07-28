"""LeWM: ViT encoder -> single 192-dim latent per frame, AdaLN-conditioned
causal predictor. Faithful to the official architecture (lucas-maes/le-wm,
arXiv:2603.19312), scaled to 64x64 inputs — deviations in lewm/README.md.

    frames (B,T,3,64,64) --ViT+CLS--> (B,T,192) --projector--> emb (B,T,192)
    actions (B,T,A)      --ActionEncoder-->                 act_emb (B,T,192)
    emb[:, :T-1], act_emb[:, :T-1] --Predictor+pred_proj--> pred (B,T-1,192)

Two structural details that matter:

  * The projector ends the encoder with **BatchNorm, not LayerNorm**: the
    paper notes a final LayerNorm pins embeddings to a sphere, which SIGReg
    cannot push toward N(0, I). The BN projector frees the latent's scale.

  * Actions are injected by **AdaLN-zero** (DiT-style), not action tokens:
    the action embedding modulates every block's normalization. Zero-init
    makes the action pathway a no-op at initialization — the block returns
    its input exactly — so conditioning fades in smoothly during training.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

LATENT_DIM = 192


# ---------------------------------------------------------------- encoder --

class ViTEncoder(nn.Module):
    """ViT-Tiny-style encoder, from scratch, CLS output.
    Official: patch 14 @ 224px, 12 layers. Here: patch 8 @ 64px, 6 layers."""

    def __init__(self, image_size: int = 64, patch: int = 8, dim: int = LATENT_DIM,
                 depth: int = 6, heads: int = 3):
        super().__init__()
        n = (image_size // patch) ** 2
        self.patch = nn.Conv2d(3, dim, kernel_size=patch, stride=patch)
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos = nn.Parameter(torch.randn(1, n + 1, dim) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=heads, dim_feedforward=4 * dim,
            activation="gelu", batch_first=True, norm_first=True, dropout=0.0,
        )
        self.blocks = nn.TransformerEncoder(layer, num_layers=depth,
                                            enable_nested_tensor=False)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B,3,H,W) -> (B,dim)
        tok = self.patch(x).flatten(2).transpose(1, 2)
        tok = torch.cat([self.cls.expand(len(tok), -1, -1), tok], dim=1)
        tok = self.blocks(tok + self.pos)
        return self.norm(tok)[:, 0]


def mlp_projector(dim: int = LATENT_DIM, hidden: int = 2048) -> nn.Sequential:
    """Linear -> BatchNorm1d -> GELU -> Linear (official `MLP`). Used after
    the encoder (projector) and after the predictor (pred_proj)."""
    return nn.Sequential(
        nn.Linear(dim, hidden), nn.BatchNorm1d(hidden), nn.GELU(),
        nn.Linear(hidden, dim),
    )


class ActionEncoder(nn.Module):
    """Action (block) -> 192-dim conditioning vector (official `Embedder`:
    bottleneck then MLP). With frame-skip action blocks on harder tasks,
    action_dim becomes frameskip * raw_dim."""

    def __init__(self, action_dim: int = 2, dim: int = LATENT_DIM, bottleneck: int = 10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(action_dim, bottleneck),
            nn.Linear(bottleneck, 4 * dim), nn.SiLU(),
            nn.Linear(4 * dim, dim),
        )

    def forward(self, a: torch.Tensor) -> torch.Tensor:
        return self.net(a)


# -------------------------------------------------------------- predictor --

class CausalAttention(nn.Module):
    def __init__(self, dim: int, heads: int = 8, dim_head: int = 32, dropout: float = 0.1):
        super().__init__()
        inner = heads * dim_head
        self.heads = heads
        self.qkv = nn.Linear(dim, 3 * inner, bias=False)
        self.out = nn.Linear(inner, dim)
        self.dropout = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q, k, v = (y.view(b, t, self.heads, -1).transpose(1, 2) for y in (q, k, v))
        o = F.scaled_dot_product_attention(
            q, k, v, is_causal=True,
            dropout_p=self.dropout if self.training else 0.0,
        )
        return self.out(o.transpose(1, 2).reshape(b, t, -1))


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, dim), nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale) + shift


class ConditionalBlock(nn.Module):
    """Transformer block with AdaLN-zero action conditioning (DiT,
    arXiv:2212.09748). The action embedding produces 6 modulation tensors
    (shift/scale/gate for attention and MLP); the zero-initialized head makes
    both residual branches vanish at init, so the block starts as identity."""

    def __init__(self, dim: int, heads: int = 8, dim_head: int = 32,
                 mlp_dim: int = 768, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn = CausalAttention(dim, heads, dim_head, dropout)
        self.mlp = FeedForward(dim, mlp_dim, dropout)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """x: (B, T, dim) latent tokens; c: (B, T, dim) action embeddings."""
        s1, sc1, g1, s2, sc2, g2 = self.adaLN_modulation(c).chunk(6, dim=-1)
        x = x + g1 * self.attn(modulate(self.norm1(x), s1, sc1))
        x = x + g2 * self.mlp(modulate(self.norm2(x), s2, sc2))
        return x


class Predictor(nn.Module):
    """Causal transformer over the frame-latent sequence.
    Official: depth 6, heads 16x64, mlp 2048. Here: depth 6, heads 8x32, mlp 768."""

    def __init__(self, dim: int = LATENT_DIM, depth: int = 6, num_frames: int = 3,
                 heads: int = 8, dim_head: int = 32, mlp_dim: int = 768,
                 dropout: float = 0.1):
        super().__init__()
        self.num_frames = num_frames
        self.pos = nn.Parameter(torch.randn(1, num_frames, dim))
        self.blocks = nn.ModuleList(
            ConditionalBlock(dim, heads, dim_head, mlp_dim, dropout)
            for _ in range(depth)
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, emb: torch.Tensor, act_emb: torch.Tensor) -> torch.Tensor:
        """(B, T<=num_frames, dim) x2 -> (B, T, dim); output token t is the
        prediction of frame t+1 (causal: it saw frames 0..t, actions 0..t)."""
        x = emb + self.pos[:, : emb.size(1)]
        for blk in self.blocks:
            x = blk(x, act_emb)
        return self.norm(x)


# ------------------------------------------------------------------ LeWM ---

class LeWM(nn.Module):
    def __init__(self, image_size: int = 64, action_dim: int = 2,
                 history: int = 3, dropout: float = 0.1):
        super().__init__()
        self.history = history
        self.encoder = ViTEncoder(image_size)
        self.projector = mlp_projector()
        self.action_encoder = ActionEncoder(action_dim)
        self.predictor = Predictor(num_frames=history, dropout=dropout)
        self.pred_proj = mlp_projector()

    def encode(self, obs: torch.Tensor, actions: torch.Tensor):
        """obs (B, T, 3, H, W), actions (B, T, A) -> emb, act_emb (B, T, 192).
        Time folds into batch for the BN projector (as in the official code)."""
        b, t = obs.shape[:2]
        z = self.encoder(obs.flatten(0, 1))
        z = self.projector(z).view(b, t, -1)
        return z, self.action_encoder(actions)

    def predict(self, ctx_emb: torch.Tensor, ctx_act: torch.Tensor) -> torch.Tensor:
        """(B, T, 192) x2 -> (B, T, 192) next-step predictions (shifted)."""
        b, t = ctx_emb.shape[:2]
        h = self.predictor(ctx_emb, ctx_act)
        return self.pred_proj(h.flatten(0, 1)).view(b, t, -1)

    @torch.no_grad()
    def rollout(self, ctx_emb: torch.Tensor, ctx_act_emb: torch.Tensor,
                future_act_emb: torch.Tensor) -> torch.Tensor:
        """Autoregressive latent rollout for planning.

        ctx_emb        (B, H, 192)  encoded context frames (H = history)
        ctx_act_emb    (B, H, 192)  embeddings of actions taken at them (the
                                    last is a placeholder — it is replaced by
                                    each imagined action in turn)
        future_act_emb (B, K, 192)  candidate future actions to simulate
        returns        (B, K, 192)  predicted latents, one per future action
        """
        embs = ctx_emb
        acts = ctx_act_emb
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
