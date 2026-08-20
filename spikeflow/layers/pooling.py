"""Spike-aware pooling, dropout, and voting layers."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class SpikingMaxPool2d(nn.Module):
    """MaxPool2d that propagates max membrane potential."""

    def __init__(self, kernel_size: int = 2, stride: int | None = None, padding: int = 0):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size, stride=stride, padding=padding)

    def forward(self, x: Tensor) -> Tensor:
        return self.pool(x)

    def reset_state(self):
        pass

    def set_sfa_mode(self, enabled: bool):
        pass

    def set_readout(self, enabled: bool):
        pass


class SpikingAvgPool2d(nn.Module):
    """AvgPool2d that averages membrane potentials."""

    def __init__(self, kernel_size: int = 2, stride: int | None = None, padding: int = 0):
        super().__init__()
        self.pool = nn.AvgPool2d(kernel_size, stride=stride, padding=padding)

    def forward(self, x: Tensor) -> Tensor:
        return self.pool(x)

    def reset_state(self):
        pass

    def set_sfa_mode(self, enabled: bool):
        pass

    def set_readout(self, enabled: bool):
        pass


class SpikingDropout(nn.Module):
    """Dropout that is aware of spike states (drops entire neurons, not individual spikes)."""

    def __init__(self, p: float = 0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        if not self.training:
            return x
        mask = (torch.rand_like(x) > self.p).float()
        return x * mask / (1 - self.p)

    def reset_state(self):
        pass

    def set_sfa_mode(self, enabled: bool):
        pass

    def set_readout(self, enabled: bool):
        pass


class VotingLayer(nn.Module):
    """Classification by majority vote over T timesteps.

    Accumulates spikes over time and returns the class with highest count.
    Used as the final layer in T-step inference.
    """

    def __init__(self, num_classes: int):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, x: Tensor) -> Tensor:
        # x: (T, batch, num_classes) or already accumulated
        if x.ndim == 3:
            return x.sum(dim=0).argmax(dim=-1)
        return x.argmax(dim=-1)

    def reset_state(self):
        pass

    def set_sfa_mode(self, enabled: bool):
        pass

    def set_readout(self, enabled: bool):
        pass
