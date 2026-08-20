"""Spiking Feed-Forward Network (FFN) — same structure as Transformer FFN."""

from __future__ import annotations

import torch.nn as nn
from torch import Tensor

from spikeflow.layers.linear import SpikingLinear


class SpikingFFN(nn.Module):
    """Spiking Feed-Forward Network.

    Architecture: Linear -> Activation -> Dropout -> Linear -> LIF
    Equivalent to Transformer's FFN but with spiking neurons.

    Uses SpikingLinear which internally has LIF neurons.
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int | None = None,
        dropout: float = 0.1,
        threshold: float = 1.0,
        tau: float = 2.0,
    ):
        super().__init__()
        d_ff = d_ff or d_model * 4  # Standard 4x expansion

        self.w1 = SpikingLinear(d_model, d_ff, threshold=threshold, tau=tau)
        self.w2 = SpikingLinear(d_ff, d_model, threshold=threshold, tau=tau)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        x = self.w1(x)
        x = self.dropout(x)
        x = self.w2(x)
        return x

    def reset_state(self):
        self.w1.reset_state()
        self.w2.reset_state()

    def set_sfa_mode(self, enabled: bool):
        self.w1.set_sfa_mode(enabled)
        self.w2.set_sfa_mode(enabled)

    def extra_repr(self) -> str:
        return f"d_model={self.w1.linear.out_features}, sfa={self.w1.neuron._sfa_mode}"
