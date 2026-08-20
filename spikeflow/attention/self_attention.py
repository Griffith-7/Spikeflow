"""Spiking Self-Attention with XNOR-based spike attention."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from spikeflow.neurons.lif import LIFNode


class SpikingSelfAttention(nn.Module):
    """Self-attention mechanism using spike-based computation.

    Instead of expensive floating-point Q*K^T attention, this uses
    XNOR-based binary attention: dot products become popcount operations.

    Standard attention:  softmax(Q @ K^T / sqrt(d)) @ V     → O(d^2) MACs
    Spike attention:    XNOR(Q, K) -> popcount -> scale @ V  → O(d^2) ADDs

    Supports three modes:
    1. 'xnor'    - Binary XNOR attention (fastest, lowest energy)
    2. 'rate'    - Rate-coded attention (better accuracy)
    3. 'hybrid'  - XNOR with rate-coded correction (best balance)
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.0,
        attention_mode: str = "xnor",
        threshold: float = 1.0,
        tau: float = 2.0,
    ):
        super().__init__()
        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.attention_mode = attention_mode
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

    def _xnor_attention(self, q: Tensor, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        """Binary XNOR attention — addition-only, no multiplications.

        Q, K are binary spikes: XNOR produces 1 when equal, 0 when different.
        Popcount(Q == K) approximates the attention weights.
        """
        # Binarize: sign(x) -> {-1, +1}, then XNOR = sign(q) * sign(k)
        q_bin = q.sign()
        k_bin = k.sign()

        # XNOR = element-wise multiply of sign representations
        # Popcount along d_head: sum of XNOR / d_head gives similarity
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
        """XNOR for coarse attention + rate-coded fine adjustment."""
        q_bin = q.sign()
        k_bin = k.sign()

        # Coarse: XNOR
        coarse_attn = torch.matmul(q_bin, k_bin.transpose(-2, -1)) / self.scale

        # Fine: rate-coded correction
        q_rate = q.float()
        k_rate = k.float()
        fine_attn = torch.matmul(q_rate, k_rate.transpose(-2, -1)) / self.scale

        # Blend: 70% coarse (fast) + 30% fine (accurate)
        attn = 0.7 * coarse_attn + 0.3 * fine_attn
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
