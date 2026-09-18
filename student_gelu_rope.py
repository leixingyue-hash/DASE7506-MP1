"""
DASE7506 MP1 - Mechanism Ablation
GELU + RoPE

Purpose:
    Ablate SwiGLU from the current SwiGLU + RoPE model.

Compared with the current best model:
    SwiGLU + RoPE  ->  GELU + RoPE

Training budget and other architectural settings are kept fixed.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Rotary Positional Embedding
# ============================================================

class RotaryEmbedding(nn.Module):

    def __init__(self, head_dim: int, context: int, base: float = 10000.0):
        super().__init__()

        if head_dim % 2 != 0:
            raise ValueError("RoPE requires an even head dimension.")

        inv_freq = 1.0 / (
            base ** (
                torch.arange(
                    0,
                    head_dim,
                    2,
                    dtype=torch.float32
                ) / head_dim
            )
        )

        positions = torch.arange(
            context,
            dtype=torch.float32
        )

        freqs = torch.outer(
            positions,
            inv_freq
        )

        self.register_buffer(
            "cos_cached",
            freqs.cos(),
            persistent=False
        )

        self.register_buffer(
            "sin_cached",
            freqs.sin(),
            persistent=False
        )

    def forward(self, x):
        """
        x: [B, H, T, D]
        """

        T = x.size(-2)

        cos = self.cos_cached[:T]
        sin = self.sin_cached[:T]

        cos = cos[None, None, :, :].to(
            device=x.device,
            dtype=x.dtype
        )

        sin = sin[None, None, :, :].to(
            device=x.device,
            dtype=x.dtype
        )

        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]

        rotated_even = (
            x_even * cos
            - x_odd * sin
        )

        rotated_odd = (
            x_even * sin
            + x_odd * cos
        )

        return torch.stack(
            (rotated_even, rotated_odd),
            dim=-1
        ).flatten(-2)


# ============================================================
# Transformer Block
# ============================================================

class StudentBlock(nn.Module):

    def __init__(
        self,
        width: int,
        n_heads: int,
        context: int
    ):
        super().__init__()

        if width % n_heads != 0:
            raise ValueError(
                "width must be divisible by n_heads"
            )

        self.n_heads = n_heads
        self.head_dim = width // n_heads

        # Same Pre-LayerNorm structure
        self.ln1 = nn.LayerNorm(width)
        self.ln2 = nn.LayerNorm(width)

        # Same attention projections
        self.qkv = nn.Linear(
            width,
            3 * width
        )

        self.proj = nn.Linear(
            width,
            width
        )

        # ----------------------------------------------------
        # ABLATION:
        # SwiGLU has been removed.
        #
        # Restore baseline GELU FFN:
        # width -> 4*width -> width
        # ----------------------------------------------------
        self.mlp = nn.Sequential(
            nn.Linear(width, 4 * width),
            nn.GELU(),
            nn.Linear(4 * width, width)
        )

        self.rope = RotaryEmbedding(
            head_dim=self.head_dim,
            context=context
        )

    def forward(self, x):

        # ----------------------------------------------------
        # Pre-LN causal self-attention
        # ----------------------------------------------------

        h = self.ln1(x)

        q, k, v = self.qkv(h).chunk(
            3,
            dim=-1
        )

        B, T, C = q.shape

        H = self.n_heads
        D = self.head_dim

        q = q.view(
            B, T, H, D
        ).transpose(1, 2)

        k = k.view(
            B, T, H, D
        ).transpose(1, 2)

        v = v.view(
            B, T, H, D
        ).transpose(1, 2)

        # Apply RoPE to Q and K
        q = self.rope(q)
        k = self.rope(k)

        attn = F.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=True
        )

        attn = (
            attn
            .transpose(1, 2)
            .contiguous()
            .view(B, T, C)
        )

        x = x + self.proj(attn)

        # GELU FFN
        x = x + self.mlp(
            self.ln2(x)
        )

        return x


# ============================================================
# Student GPT
# ============================================================

class StudentGPT(nn.Module):

    def __init__(
        self,
        vocab_size: int,
        width: int,
        n_heads: int,
        depth: int,
        context: int
    ):
        super().__init__()

        self.context = context
        self.vocab_size = vocab_size

        # Token embedding only.
        # Position is represented by RoPE.
        self.tok_emb = nn.Embedding(
            vocab_size,
            width
        )

        self.blocks = nn.ModuleList(
            [
                StudentBlock(
                    width=width,
                    n_heads=n_heads,
                    context=context
                )
                for _ in range(depth)
            ]
        )

        self.ln_f = nn.LayerNorm(width)

        self.head = nn.Linear(
            width,
            vocab_size,
            bias=False
        )

        # Weight tying
        self.head.weight = self.tok_emb.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):

        if isinstance(module, nn.Linear):

            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=0.02
            )

            if module.bias is not None:
                nn.init.zeros_(
                    module.bias
                )

        elif isinstance(module, nn.Embedding):

            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=0.02
            )

    def features(self, input_ids):

        B, T = input_ids.shape

        if T > self.context:
            raise ValueError(
                f"Sequence length {T} exceeds "
                f"context length {self.context}"
            )

        # No learned absolute position embedding
        x = self.tok_emb(input_ids)

        for block in self.blocks:
            x = block(x)

        return self.ln_f(x)

    def forward(self, input_ids):

        return self.head(
            self.features(input_ids)
        )

    @torch.no_grad()
    def predict_log_probs(self, input_ids):

        logits = self.forward(input_ids)

        return F.log_softmax(
            logits,
            dim=-1
        )

    def reset_state(self):
        # No state is carried across evaluation windows.
        pass


# ============================================================
# Coursework factory
# ============================================================

def build_model(config):

    return StudentGPT(
        vocab_size=config["vocab"],
        width=config["width"],
        n_heads=config["heads"],
        depth=config["depth"],
        context=config["context"]
    )