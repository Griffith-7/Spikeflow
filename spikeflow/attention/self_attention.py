"""Spiking Self-Attention with XNOR-based spike attention."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from spikeflow.neurons.lif import LIFNode


class SpikingSelfAttention(nn.Module):
    """Self-attention mechanism using spike-based computation.

    Q and K come from spiking neurons; attention logits use their (near-)
    binary activations. The ``xnor``/``sign`` mode computes
    ``sign(Q) @ sign(K)^T``, which is mathematically equivalent to an
    XNOR + popcount similarity: matching bits contribute +1, mismatching
    bits -1. Note that on current hardware this is still *executed* as a
    floating-point matmul — the addition-only speedup of true bit-packed
    XNOR-popcount requires custom kernels, so treat this mode as a
    functional simulation of binary attention, not a faster matmul.

    Supports three modes:
    1. 'xnor'/'sign' - binarized QK^T via sign (simulated popcount)
    2. 'rate'        - Rate-coded attention (better accuracy)
    3. 'hybrid'      - blend of the two (best balance)
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.0,
        attention_mode: str = "xnor",
        threshold: float = 1.0,
        tau: float = 2.0,
        hybrid_blend: float = 0.7,
    ):
        super().__init__()
        assert d_model % n_heads == 0
        valid_modes = ("xnor", "sign", "rate", "hybrid")
        if attention_mode not in valid_modes:
            raise ValueError(
                f"attention_mode must be one of {valid_modes}, got '{attention_mode}'"
            )
        if not 0.0 <= hybrid_blend <= 1.0:
            raise ValueError("hybrid_blend must be between 0.0 and 1.0")

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        if attention_mode == "sign":
            attention_mode = "xnor"  # alias
        self.attention_mode = attention_mode
        self.hybrid_blend = hybrid_blend
        self.scale = self.d_head ** 0.5

        # Standard linear projections (same as nn.MultiheadAttention)
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        # Spiking neurons after projections
        self.q_neuron = LIFNode(threshold=threshold, tau=tau)
        self.k_neuron = LIFNode(threshold=threshold, tau=tau)
        self.v_neuron = LIFNode(threshold=threshold, tau=tau)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: Tensor,
        mask: Tensor | None = None,
        return_attention: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        B, N, _ = x.shape

        # Project to Q, K, V
        q = self.q_neuron(self.q_proj(x).reshape(B, N, self.n_heads, self.d_head).transpose(1, 2))
        k = self.k_neuron(self.k_proj(x).reshape(B, N, self.n_heads, self.d_head).transpose(1, 2))
        v = self.v_neuron(self.v_proj(x).reshape(B, N, self.n_heads, self.d_head).transpose(1, 2))

        if self.attention_mode == "xnor":
            attn_out, attn_weights = self._xnor_attention(q, k, v)
        elif self.attention_mode == "rate":
            attn_out, attn_weights = self._rate_attention(q, k, v)
        else:
            attn_out, attn_weights = self._hybrid_attention(q, k, v)

        # Reshape back
        attn_out = attn_out.transpose(1, 2).reshape(B, N, self.d_model)
        output = self.out_proj(attn_out)

        if return_attention:
            return output, attn_weights
        return output

    def _binarize(self, x: Tensor) -> Tensor:
        """Map activations to {-1, +1}.

        ``sign()`` alone would emit 0 for exact zeros, which breaks the
        bipolar assumption of XNOR-style similarity — ties are resolved to +1.
        """
        one = torch.ones_like(x)
        return torch.where(x >= 0, one, -one)

    def _xnor_attention(self, q: Tensor, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        """Binarized (sign) attention — functional simulation of XNOR-popcount.

        With bipolar {-1, +1} codes, elementwise product == XNOR: +1 when bits
        match, -1 when they differ. The row sum is therefore
        popcount(match) - popcount(mismatch), i.e. the XNOR similarity score.
        Executed here as an fp32 matmul; a bit-packed integer kernel would be
        needed for the actual addition-only speedup.
        """
        q_bin = self._binarize(q)
        k_bin = self._binarize(k)

        attn = torch.matmul(q_bin, k_bin.transpose(-2, -1)) / self.scale

        # Softmax to get attention weights
        attn_weights = F.softmax(attn, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Use rate-coded values for V (not binary) to preserve information
        attn_out = torch.matmul(attn_weights, v)
        return attn_out, attn_weights

    def _rate_attention(self, q: Tensor, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        """Standard attention but with spike-rate values."""
        attn = torch.matmul(q.float(), k.float().transpose(-2, -1)) / self.scale
        attn_weights = F.softmax(attn, dim=-1)
        attn_weights = self.dropout(attn_weights)
        attn_out = torch.matmul(attn_weights, v.float())
        return attn_out, attn_weights

    def _hybrid_attention(self, q: Tensor, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        """Binarized coarse attention + rate-coded fine adjustment."""
        q_bin = self._binarize(q)
        k_bin = self._binarize(k)

        # Coarse: XNOR-style similarity
        coarse_attn = torch.matmul(q_bin, k_bin.transpose(-2, -1)) / self.scale

        # Fine: rate-coded correction
        q_rate = q.float()
        k_rate = k.float()
        fine_attn = torch.matmul(q_rate, k_rate.transpose(-2, -1)) / self.scale

        # Blend: coarse (fast) + fine (accurate)
        attn = self.hybrid_blend * coarse_attn + (1.0 - self.hybrid_blend) * fine_attn
        attn_weights = F.softmax(attn, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_out = torch.matmul(attn_weights, v.float())
        return attn_out, attn_weights

    def reset_state(self):
        self.q_neuron.reset_state()
        self.k_neuron.reset_state()
        self.v_neuron.reset_state()

    def set_sfa_mode(self, enabled: bool):
        self.q_neuron.set_sfa_mode(enabled)
        self.k_neuron.set_sfa_mode(enabled)
        self.v_neuron.set_sfa_mode(enabled)

    def extra_repr(self) -> str:
        return (
            f"d_model={self.d_model}, n_heads={self.n_heads}, "
            f"mode={self.attention_mode}, sfa={self.q_neuron._sfa_mode}"
        )
