"""Integrate-and-Fire (IF) neuron model."""

from __future__ import annotations

import torch

from spikeflow.neurons.base import BaseNeuron


class IFNode(BaseNeuron):
    """Integrate-and-Fire neuron (no leak).

    v(t+1) = v(t) + x(t)
    spike = (v >= threshold)
    if spike: v = v_reset

    Simplest spiking neuron. Good for SFA training as it
    reduces to ReLU when threshold=1 and no reset.
    """

    def __init__(
        self,
        threshold: float = 1.0,
        surrogate: str = "sigmoid",
        v_reset: float = 0.0,
    ):
        super().__init__(threshold, surrogate, tau=float("inf"), dt=1.0, v_reset=v_reset)

    def charge(self, x: torch.Tensor) -> torch.Tensor:
        if self._sfa_mode:
            return torch.relu(x)
        if self._readout:
            self.v = torch.relu(self.v + x).clamp(max=100.0)
            return self.v
        self.v = self.v + x
        return self.v

    def extra_repr(self) -> str:
        return f"IFNode(threshold={self.threshold_module.threshold}, sfa={self._sfa_mode})"
