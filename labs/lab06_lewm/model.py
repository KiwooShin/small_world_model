"""LeWM model: ViT encoder -> 192-dim latent, AdaLN-conditioned predictor.

Faithful to the official architecture (lucas-maes/le-wm, arXiv:2603.19312),
scaled to 64x64 PushWorld — deviations are listed in the README table.

The pipeline per training window of T frames:

    frames (B,T,3,64,64) --ViT+CLS--> (B,T,192) --projector--> emb (B,T,192)
    actions (B,T,2)      --ActionEncoder-->                 act_emb (B,T,192)
    emb[:, :T-1], act_emb[:, :T-1] --Predictor+pred_proj--> pred (B,T-1,192)

Two structural details matter more than they look:

  * The projector ends the encoder with **BatchNorm, not LayerNorm**. The
    paper is explicit about why: the ViT's final LayerNorm confines
    embeddings to a sphere, and you cannot push a sphere toward N(0, I) —
    SIGReg "cannot be optimized effectively" through it. The BN projector
    gives the latent an unconstrained scale.

  * The predictor conditions on actions via **AdaLN-zero** (your TASK 2),
    not by concatenating action tokens. Zero-init makes the action pathway
    a no-op at initialization, so early training learns unconditional
    dynamics first and conditioning fades in smoothly.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

LATENT_DIM = 192


# ---------------------------------------------------------------- encoder --

class ViTEncoder(nn.Module):
    """ViT-Tiny-style encoder, from scratch (no pretraining), CLS output.

    Official LeWM: ViT-Tiny, patch 14 @ 224px, 12 layers, 3 heads.
    Lab scale:     patch 8 @ 64px (-> 64 tokens), 6 layers, 3 heads.
    """

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
        tok = self.patch(x).flatten(2).transpose(1, 2)          # (B, N, dim)
        tok = torch.cat([self.cls.expand(len(tok), -1, -1), tok], dim=1)
        tok = self.blocks(tok + self.pos)
        return self.norm(tok)[:, 0]                              # CLS token


def mlp_projector(dim: int = LATENT_DIM, hidden: int = 2048) -> nn.Sequential:
    """Linear -> BatchNorm1d -> GELU -> Linear, as in the official code.
    Used twice: after the encoder (projector) and after the predictor
    (pred_proj). See module docstring for why BN and not LN."""
    return nn.Sequential(
        nn.Linear(dim, hidden), nn.BatchNorm1d(hidden), nn.GELU(),
        nn.Linear(hidden, dim),
    )


class ActionEncoder(nn.Module):
    """Action block -> 192-dim conditioning vector (official `Embedder`:
    a small bottleneck then an MLP). action_dim=2 here; with frame-skip on
    MuJoCo data later, action_dim becomes frameskip * raw_dim."""

    def __init__(self, action_dim: int = 2, dim: int = LATENT_DIM, bottleneck: int = 10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(action_dim, bottleneck),
            nn.Linear(bottleneck, 4 * dim), nn.SiLU(),
            nn.Linear(4 * dim, dim),
        )

    def forward(self, a: torch.Tensor) -> torch.Tensor:  # (..., A) -> (..., dim)
        return self.net(a)


# -------------------------------------------------------------- predictor --

class CausalAttention(nn.Module):
    """Multi-head self-attention with a causal mask (SDPA). Provided."""

    def __init__(self, dim: int, heads: int = 8, dim_head: int = 32, dropout: float = 0.1):
        super().__init__()
        inner = heads * dim_head
        self.heads = heads
        self.qkv = nn.Linear(dim, 3 * inner, bias=False)
        self.out = nn.Linear(inner, dim)
        self.dropout = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, T, dim)
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


class ConditionalBlock(nn.Module):
    """Transformer block with AdaLN-zero action conditioning.   ★ TASK 2 ★

    This is the DiT conditioning mechanism (Peebles & Xie 2023), reused by
    LeWM to inject actions: instead of adding action tokens to the sequence,
    the action embedding *modulates* the normalization of every block.

    The recipe — implement `__init__` additions and `forward`:

    1. Two LayerNorms WITHOUT learnable affine params
       (`elementwise_affine=False`, eps 1e-6) — provided below. Their scale
       and shift come from the action instead, which is the whole idea.

    2. An `adaLN_modulation` head: SiLU followed by a Linear from `dim` to
       **6 * dim**. Its output, given the action embedding c (B, T, dim),
       is chunked into six (B, T, dim) tensors:
           shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp

    3. ZERO-initialize that Linear's weight AND bias. Consequence you can
       check by hand: all six chunks are zero, so with
           modulate(h, shift, scale) = h * (1 + scale) + shift
       the modulation is the identity and the gates kill both residual
       branches — the block returns its input EXACTLY at init, whatever the
       action says. check.py asserts this to float precision.

    4. Forward:
           x = x + gate_msa * attn(modulate(norm1(x), shift_msa, scale_msa))
           x = x + gate_mlp * mlp (modulate(norm2(x), shift_mlp, scale_mlp))

    Think about why the gate (not just zero shift/scale) is needed for the
    exact-identity property — attn of a normalized input is not zero.
    """

    def __init__(self, dim: int, heads: int = 8, dim_head: int = 32,
                 mlp_dim: int = 768, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn = CausalAttention(dim, heads, dim_head, dropout)
        self.mlp = FeedForward(dim, mlp_dim, dropout)
        # TODO(TASK 2): build self.adaLN_modulation and zero-init its Linear.
        raise NotImplementedError("TASK 2: ConditionalBlock.__init__ AdaLN head")

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """x: (B, T, dim) latent tokens; c: (B, T, dim) action embeddings."""
        raise NotImplementedError("TASK 2: ConditionalBlock.forward")


class Predictor(nn.Module):
    """Causal transformer over the frame-latent sequence. Provided —
    it just stacks your ConditionalBlocks.

    Official: depth 6, heads 16, head_dim 64, mlp 2048.
    Lab:      depth 6, heads 8,  head_dim 32, mlp 768.
    """

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

        BatchNorm detail: the projector expects a flat batch, so time is
        folded into batch and unfolded after — meaning BN statistics mix all
        time steps, which is fine (and is what the official code does)."""
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
        """Autoregressive latent rollout for planning. Provided.

        ctx_emb        (B, H, 192)  encoded context frames (H = history)
        ctx_act_emb    (B, H, 192)  embeddings of the actions taken at them
        future_act_emb (B, K, 192)  candidate future actions to simulate
        returns        (B, K, 192)  predicted latents, one per future action

        Note the action alignment: when predicting from the window's last
        frame, the action *at* that frame is the candidate being evaluated —
        so the action window is the context actions shifted by the number of
        steps already imagined.
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
            pred = self.predict(window_e, window_a)[:, -1:]      # newest step
            embs = torch.cat([embs, pred], dim=1)
            out.append(pred)
        return torch.cat(out, dim=1)
