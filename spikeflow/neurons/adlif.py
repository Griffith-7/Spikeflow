"""Adaptive Leaky Integrate-and-Fire (ALIF) neuron model."""

from __future__ import annotations

import torch

from spikeflow.neurons.base import BaseNeuron


class AdaptiveLIFNode(BaseNeuron):
    """Adaptive LIF neuron with learnable threshold adaptation.

    Adds a slow adaptation variable a that increases the effective
    threshold after each spike, modeling spike-frequency adaptation:

        v(t+1) = decay * v(t) + x(t) - a(t)
        spike  = (v >= threshold)
        a(t+1) = decay_a * a(t) + delta_a * spike

    (The adaptation variable is subtracted from the membrane charge and the
    neuron still fires at ``threshold`` — the two formulations are equivalent
    up to a constant shift of v.)

    Useful for detecting temporal changes and encoding novelty.
    """

    def __init__(
        self,
        threshold: float = 1.0,
        surrogate: str = "sigmoid",
        tau: float = 2.0,
        tau_adapt: float = 50.0,
        delta_a: float = 0.05,
        dt: float = 1.0,
        v_reset: float = 0.0,
    ):
        super().__init__(threshold, surrogate, tau, dt, v_reset)
        self.register_buffer("decay", torch.exp(torch.tensor(-dt / tau)))
        self.register_buffer("decay_a", torch.exp(torch.tensor(-dt / tau_adapt)))
        self.delta_a = delta_a
        self.a: torch.Tensor | None = None

    def reset_state(self):
        super().reset_state()
        self.a = None

    def charge(self, x: torch.Tensor) -> torch.Tensor:
        if self._sfa_mode:
            return torch.relu(x)
        if self._readout:
            self.v = torch.relu(self.v * self.decay + x - self.a).clamp(max=100.0)
            self.a = self.a * self.decay_a
            return self.v
        self.v = self.v * self.decay + x - self.a
        self.a = self.a * self.decay_a
        return self.v

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.v is None:
            self.v = torch.zeros_like(x)
        if self.a is None:
            self.a = torch.zeros_like(x)

        self.v = self.charge(x)
        if self._readout or self._sfa_mode:
            # Readout/SFA mode: no binary spikes, so derive pseudo-spikes from
            # threshold crossings to keep the adaptation variable updating.
            pseudo = (self.v >= self.threshold_module.threshold).detach().to(self.a.dtype)
            self.a = self.a + self.delta_a * pseudo
            return self.v
        spike = self.fire(self.v)
        self.neuronal_reset(spike)
        self.a = self.a + self.delta_a * spike.detach()
        return spike

    def extra_repr(self) -> str:
        base = super().extra_repr()
        return f"AdaptiveLIFNode({base}, delta_a={self.delta_a})"
