"""Izhikevich neuron model."""

from __future__ import annotations

import torch

from spikeflow.neurons.base import BaseNeuron


class IzhikevichNode(BaseNeuron):
    """Izhikevich spiking neuron with rich dynamics.

    Models: RS, IB, CH, LTS spiking patterns via (a, b, c, d) params.
        v' = 0.04*v^2 + 5*v + 140 - u + I
        u' = a*(b*v - u)
        if v >= 30: v = c, u = u + d

    Supports regular spiking (RS), intrinsically bursting (IB),
    chattering (CH), and low-threshold spiking (LTS).
    """

    PRESETS = {
        "rs":  (0.02, 0.2,  -65.0, 8.0),
        "ib":  (0.02, 0.2,  -55.0, 4.0),
        "ch":  (0.02, 0.2,  -50.0, 2.0),
        "lts": (0.02, 0.25, -65.0, 2.0),
    }

    def __init__(
        self,
        preset: str = "rs",
        a: float = 0.02,
        b: float = 0.2,
        c: float = -65.0,
        d: float = 8.0,
        surrogate: str = "sigmoid",
        v_thresh: float = 30.0,
    ):
        super().__init__(threshold=v_thresh, surrogate=surrogate, tau=float("inf"), dt=1.0)
        if preset in self.PRESETS:
            a, b, c, d = self.PRESETS[preset]
        self.izh_a = a
        self.izh_b = b
        self.izh_c = c
        self.izh_d = d
        self.v_thresh = v_thresh
        self.u: torch.Tensor | None = None

    def reset_state(self):
        super().reset_state()
        self.u = None

    def charge(self, x: torch.Tensor) -> torch.Tensor:
        if self._sfa_mode:
            return torch.relu(x)
        if self._readout:
            self.v = torch.relu(
                self.v + 0.04 * self.v.pow(2) + 5 * self.v + 140 - self.u + x
            ).clamp(max=100.0)
            self.u = self.u + self.izh_a * (self.izh_b * self.v - self.u)
            return self.v
        self.v = self.v + 0.04 * self.v.pow(2) + 5 * self.v + 140 - self.u + x
        self.u = self.u + self.izh_a * (self.izh_b * self.v - self.u)
        return self.v

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.v is None:
            self.v = torch.zeros_like(x) - 65.0
        if self.u is None:
            self.u = self.izh_b * self.v

        self.v = self.charge(x)
        if self._sfa_mode or self._readout:
            return self.v
        fired = (self.v >= self.v_thresh).float()
        self.v = self.v * (1.0 - fired.detach()) + self.izh_c * fired.detach()
        self.u = self.u + self.izh_d * fired.detach()
        return fired

    def extra_repr(self) -> str:
        base = super().extra_repr()
        return f"IzhikevichNode({base}, a={self.izh_a}, b={self.izh_b})"
