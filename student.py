"""
DASE7506 MP1 - Experiment 3
Parameter-Matched SwiGLU + RoPE

Changes relative to the supplied baseline:
1. GELU FFN -> parameter-matched SwiGLU
2. Learned absolute positional embeddings -> RoPE

RoPE is applied to query and key vectors before causal attention.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# SwiGLU
# ============================================================

class SwiGLU(nn.Module):

    def __init__(self, width: int):
        super().__init__()

        # Parameter-match approximately against baseline 4x FFN.
        hidden = int((8 * width) / 3)
        hidden = ((hidden + 7) // 8) * 8

        self.gate = nn.Linear(width, hidden)
        self.value = nn.Linear(width, hidden)
        self.down = nn.Linear(hidden, width)

    def forward(self, x):
        return self.down(
            F.silu(self.gate(x)) * self.value(x)
        )


# ============================================================
# Rotary Positional Embedding (RoPE)
# ============================================================

class RotaryEmbedding(nn.Module):

    def __init__(self, head_dim: int, context: int, base: float = 10000.0):
        super().__init__()

        if head_dim % 2 != 0:
            raise ValueError("RoPE requires an even head dimension.")

        # Frequencies:
        # theta_i = base^(-2i / head_dim)
        inv_freq = 1.0 / (
            base ** (
                torch.arange(0, head_dim, 2, dtype=torch.float32)
                / head_dim
            )
        )

        positions = torch.arange(
            context,
            dtype=torch.float32
        )

        # [context, head_dim / 2]
        freqs = torch.outer(positions, inv_freq)

        # Store as buffers so they are moved with the model,
        # but are not trainable parameters.
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
        x shape:
            [B, H, T, D]
        """

        T = x.size(-2)

        cos = self.cos_cached[:T]
        sin = self.sin_cached[:T]

        # [1, 1, T, D/2]
        cos = cos[None, None, :, :].to(
            device=x.device,
            dtype=x.dtype
        )

        sin = sin[None, None, :, :].to(
            device=x.device,
            dtype=x.dtype
        )

        # Pair adjacent dimensions:
        # (x0, x1), (x2, x3), ...
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]

        rotated_even = x_even * cos - x_odd * sin
        rotated_odd = x_even * sin + x_odd * cos

        # Interleave even/odd dimensions again
        rotated = torch.stack(
            (rotated_even, rotated_odd),
            dim=-1
        )

        return rotated.flatten(-2)


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

        self.ln1 = nn.LayerNorm(width)
        self.ln2 = nn.LayerNorm(width)

        self.qkv = nn.Linear(
            width,
            3 * width
        )

        self.proj = nn.Linear(
            width,
            width
        )

        self.mlp = SwiGLU(width)

        self.rope = RotaryEmbedding(
            head_dim=self.head_dim,
            context=context
        )

    def forward(self, x):

        # ----------------------------------------------------
        # Pre-LN causal attention
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

        # ----------------------------------------------------
        # RoPE is applied to Q and K only
        # ----------------------------------------------------

        q = self.rope(q)
        k = self.rope(k)

        # ----------------------------------------------------
        # Causal scaled dot-product attention
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # SwiGLU FFN
        # ----------------------------------------------------

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

        # Token embeddings only.
        #
        # Unlike the baseline, there is NO learned
        # absolute positional embedding here.
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
                nn.init.zeros_(module.bias)

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

        # No absolute positional embedding.
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
        # No cross-window state.
        pass


# ============================================================
# Factory required by the coursework
# ============================================================

def build_model(config):

    return StudentGPT(
        vocab_size=config["vocab"],
        width=config["width"],
        n_heads=config["heads"],
        depth=config["depth"],
        context=config["context"]
    )