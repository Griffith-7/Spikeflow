"""Spiking linear (fully connected) layer."""

from __future__ import annotations

import torch.nn as nn
from torch import Tensor

from spikeflow.neurons.lif import LIFNode


class SpikingLinear(nn.Module):
    """Drop-in replacement for nn.Linear that outputs spikes.

    Architecture: Linear -> LIF Neuron

    In SFA training mode, the LIF neuron behaves as ReLU,
    so this is equivalent to nn.Linear + ReLU during training.

    Usage:
        layer = SpikingLinear(768, 3072)
        # Training (T=1): outputs continuous activations
        # Inference (T=4): outputs binary spikes over 4 timesteps
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        threshold: float = 1.0,
        surrogate: str = "sigmoid",
        tau: float = 2.0,
        readout: bool = False,
    ):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.neuron = LIFNode(
            threshold=threshold,
            surrogate=surrogate,
            tau=tau,
        )
        if readout:
            self.neuron.set_readout(True)

    def forward(self, x: Tensor) -> Tensor:
        return self.neuron(self.linear(x))

    def reset_state(self):
        self.neuron.reset_state()

    def set_sfa_mode(self, enabled: bool):
        self.neuron.set_sfa_mode(enabled)

    def set_readout(self, enabled: bool):
        self.neuron.set_readout(enabled)

    def extra_repr(self) -> str:
        return (
            f"in={self.linear.in_features}, "
            f"out={self.linear.out_features}, "
            f"bias={self.linear.bias is not None}, "
            f"sfa={self.neuron._sfa_mode}"
        )
