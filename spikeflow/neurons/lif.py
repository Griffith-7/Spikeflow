"""Leaky Integrate-and-Fire (LIF) neuron model."""

from __future__ import annotations

import torch
import torch.nn as nn

from spikeflow.neurons.base import BaseNeuron


class LIFNode(BaseNeuron):
    """Leaky Integrate-and-Fire neuron.

    Membrane potential dynamics:
        v(t+1) = v(t) * decay + x(t)
        spike = (v >= threshold)
        if spike: v = v_reset

    Where decay = exp(-dt / tau)

    In SFA mode, behaves as: spike = ReLU(x), enabling T=1 training.
    """

    def __init__(
        self,
        threshold: float = 1.0,
        surrogate: str = "sigmoid",
        tau: float = 2.0,
        dt: float = 1.0,
        v_reset: float = 0.0,
    ):
        super().__init__(threshold, surrogate, tau, dt, v_reset)
        self.register_buffer("decay", torch.exp(torch.tensor(-dt / tau)))

    def charge(self, x: torch.Tensor) -> torch.Tensor:
        if self._sfa_mode:
            return torch.relu(x)
        if self._readout:
            self.v = torch.relu(self.v * self.decay + x).clamp(max=100.0)
            return self.v
        self.v = self.v * self.decay + x
        return self.v

    def extra_repr(self) -> str:
        base = super().extra_repr()
        return f"LIFNode({base}, decay={self.decay:.4f})"


class ParametricLIFNode(BaseNeuron):
    """Parametric LIF neuron with learnable time constant.

    The decay factor is learned per-channel, allowing different
    neurons to have different temporal dynamics.
    """

    def __init__(
        self,
        channels: int,
        threshold: float = 1.0,
        surrogate: str = "sigmoid",
        tau_init: float = 2.0,
        dt: float = 1.0,
        v_reset: float = 0.0,
    ):
        super().__init__(threshold, surrogate, tau_init, dt, v_reset)
        self.tau_param = nn.Parameter(
            torch.full((channels,), tau_init)
        )

    @property
    def decay(self) -> torch.Tensor:
        return torch.exp(-self.dt / self.tau_param)

    def charge(self, x: torch.Tensor) -> torch.Tensor:
        if self._sfa_mode:
            return torch.relu(x)
        if self._readout:
            decay = self.decay.view(1, -1, *([1] * (x.ndim - 2)))
            self.v = torch.relu(self.v * decay + x).clamp(max=100.0)
            return self.v
        decay = self.decay.view(1, -1, *([1] * (x.ndim - 2)))
        self.v = self.v * decay + x
        return self.v

    def extra_repr(self) -> str:
        base = super().extra_repr()
        return f"ParametricLIFNode({base})"
