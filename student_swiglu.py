"""
DASE7506 MP1 - Student Model
Experiment 1: Parameter-Matched SwiGLU

Main change:
    Replace the baseline GELU feed-forward network (FFN)
    with a parameter-matched SwiGLU FFN.

Everything else remains close to the supplied baseline:
    - Pre-LayerNorm
    - Causal self-attention
    - Learned positional embeddings
    - Tied token embedding / output head
    - No cross-window state
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 1. SwiGLU Feed-Forward Network
# ============================================================

class SwiGLU(nn.Module):
    """
    SwiGLU feed-forward network.

    SwiGLU(x) =
        W_down( SiLU(W_gate(x)) * W_value(x) )

    The hidden dimension is chosen to approximately match
    the parameter count of the baseline 4x GELU FFN.
    """

    def __init__(self, width: int):
        super().__init__()

        # ----------------------------------------------------
        # Parameter matching
        #
        # Baseline FFN:
        #   width -> 4*width -> width
        #
        # Approximate matrix parameter count:
        #   width * 4width + 4width * width
        #   = 8 * width^2
        #
        # SwiGLU:
        #   gate  : width -> hidden
        #   value : width -> hidden
        #   down  : hidden -> width
        #
        # Approximate matrix parameter count:
        #   3 * width * hidden
        #
        # Match them:
        #   3 * width * hidden ~= 8 * width^2
        #
        # Therefore:
        #   hidden ~= (8/3) * width
        # ----------------------------------------------------

        hidden = int((8 * width) / 3)

        # Round hidden dimension up to a multiple of 8.
        hidden = ((hidden + 7) // 8) * 8

        self.gate = nn.Linear(width, hidden)
        self.value = nn.Linear(width, hidden)
        self.down = nn.Linear(hidden, width)

    def forward(self, x):
        gate = F.silu(self.gate(x))
        value = self.value(x)

        return self.down(gate * value)


# ============================================================
# 2. Transformer Block
# ============================================================

class StudentBlock(nn.Module):
    """
    Transformer block based on the supplied baseline.

    The only main structural change is:
        GELU FFN -> SwiGLU FFN
    """

    def __init__(self, width: int, n_heads: int):
        super().__init__()

        # Same Pre-LayerNorm design as baseline
        self.ln1 = nn.LayerNorm(width)
        self.ln2 = nn.LayerNorm(width)

        # Same attention projections as baseline
        self.qkv = nn.Linear(width, 3 * width)
        self.proj = nn.Linear(width, width)

        self.n_heads = n_heads

        # Main contribution of Experiment 1
        self.mlp = SwiGLU(width)

    def forward(self, x):

        # ====================================================
        # Causal Self-Attention
        # ====================================================

        h = self.ln1(x)

        q, k, v = self.qkv(h).chunk(3, dim=-1)

        B, T, C = q.shape

        H = self.n_heads
        D = C // H

        # [B, T, C]
        # ->
        # [B, H, T, D]

        q = q.view(B, T, H, D).transpose(1, 2)
        k = k.view(B, T, H, D).transpose(1, 2)
        v = v.view(B, T, H, D).transpose(1, 2)

        # PyTorch causal scaled dot-product attention
        attn = F.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=True,
        )

        # [B, H, T, D]
        # ->
        # [B, T, C]

        attn = (
            attn
            .transpose(1, 2)
            .contiguous()
            .view(B, T, C)
        )

        # Attention residual connection
        x = x + self.proj(attn)

        # ====================================================
        # SwiGLU Feed-Forward Network
        # ====================================================

        x = x + self.mlp(self.ln2(x))

        return x


# ============================================================
# 3. Student GPT
# ============================================================

class StudentGPT(nn.Module):

    def __init__(
        self,
        vocab_size: int,
        width: int,
        n_heads: int,
        depth: int,
        context: int,
    ):
        super().__init__()

        self.context = context
        self.vocab_size = vocab_size

        # ----------------------------------------------------
        # Embeddings
        # ----------------------------------------------------

        self.tok_emb = nn.Embedding(
            vocab_size,
            width,
        )

        self.pos_emb = nn.Embedding(
            context,
            width,
        )

        # ----------------------------------------------------
        # Transformer blocks
        # ----------------------------------------------------

        self.blocks = nn.ModuleList(
            [
                StudentBlock(
                    width=width,
                    n_heads=n_heads,
                )
                for _ in range(depth)
            ]
        )

        # Final LayerNorm
        self.ln_f = nn.LayerNorm(width)

        # Output language-model head
        self.head = nn.Linear(
            width,
            vocab_size,
            bias=False,
        )

        # ----------------------------------------------------
        # Weight tying
        #
        # Same design as baseline:
        # input token embeddings and output LM head
        # share the same weight matrix.
        # ----------------------------------------------------

        self.head.weight = self.tok_emb.weight

        # Initialize model
        self.apply(self._init_weights)

    # ========================================================
    # Weight initialization
    # ========================================================

    def _init_weights(self, module):

        if isinstance(module, nn.Linear):

            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=0.02,
            )

            if module.bias is not None:
                nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):

            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=0.02,
            )

    # ========================================================
    # Feature extraction
    # ========================================================

    def features(self, input_ids):

        B, T = input_ids.shape

        if T > self.context:
            raise ValueError(
                f"Sequence length {T} exceeds "
                f"context length {self.context}"
            )

        positions = torch.arange(
            T,
            device=input_ids.device,
        )

        # Token embedding + learned positional embedding
        x = (
            self.tok_emb(input_ids)
            + self.pos_emb(positions)[None, :, :]
        )

        # Transformer blocks
        for block in self.blocks:
            x = block(x)

        # Final normalization
        x = self.ln_f(x)

        return x

    # ========================================================
    # Standard forward
    # ========================================================

    def forward(self, input_ids):

        x = self.features(input_ids)

        logits = self.head(x)

        return logits

    # ========================================================
    # Required evaluation interface
    # ========================================================

    @torch.no_grad()
    def predict_log_probs(self, input_ids):
        """
        Required by evaluate.py.

        Returns normalized natural-log probabilities
        with shape:

            [batch, sequence, vocab_size]
        """

        logits = self.forward(input_ids)

        return F.log_softmax(
            logits,
            dim=-1,
        )

    # ========================================================
    # Required state-reset interface
    # ========================================================

    def reset_state(self):
        """
        This model does not keep state between evaluation
        windows, so there is nothing to reset.
        """

        pass


# ============================================================
# 4. Factory required by train.py / evaluate.py
# ============================================================

def build_model(config):

    return StudentGPT(
        vocab_size=config["vocab"],
        width=config["width"],
        n_heads=config["heads"],
        depth=config["depth"],
        context=config["context"],
    )