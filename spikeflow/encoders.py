"""Input encoding methods for converting continuous data to spike trains.

Supports neuromorphic benchmarks where inputs must be rate-coded,
latency-coded, or population-coded over T timesteps.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class PoissonEncoder(nn.Module):
    """Rate coding: spike probability proportional to input intensity.

    For input x in [0, 1], at each timestep each neuron fires with P=x.
    Output shape: (T, *x.shape) or (*x.shape,) depending on temporal_output.
    """

    def __init__(self, timesteps: int = 4, temporal_output: bool = True):
        super().__init__()
        self.T = timesteps
        self.temporal_output = temporal_output

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.clamp(0, 1)
        if not self.temporal_output:
            return (torch.rand_like(x) < x).float()
        return (torch.rand(self.T, *x.shape, device=x.device) < x.unsqueeze(0)).float()


class LatencyEncoder(nn.Module):
    """Latency coding (TTFS): stronger inputs fire earlier.

    Input x in [0, 1] -> spike at time t = floor((1 - x) * T).
    Neuron fires once at its latency time, stays silent otherwise.
    """

    def __init__(self, timesteps: int = 4, temporal_output: bool = True):
        super().__init__()
        self.T = timesteps
        self.temporal_output = temporal_output

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.clamp(0, 1)
        spike_times = ((1 - x) * self.T).long().clamp(0, self.T - 1)
        if not self.temporal_output:
            spikes = torch.zeros_like(x)
            spikes.scatter_(0, spike_times, 1.0)
            return spikes
        out = torch.zeros(self.T, *x.shape, device=x.device)
        for t in range(self.T):
            out[t] = (spike_times == t).float()
        return out


class PopulationEncoder(nn.Module):
    """Population coding: each input value activates a Gaussian-tuned population.

    Each input value is encoded by N neurons with preferred values evenly
    spaced across [0, 1]. Spike rate = Gaussian tuning curve.
    """

    def __init__(self, n_neurons: int = 4, timesteps: int = 4, sigma: float = 0.1, temporal_output: bool = True):
        super().__init__()
        self.N = n_neurons
        self.T = timesteps
        self.sigma = sigma
        self.temporal_output = temporal_output
        self.register_buffer("preferred", torch.linspace(0, 1, n_neurons))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        x_flat = x.reshape(-1, shape[-1])
        diff = x_flat.unsqueeze(-1) - self.preferred
        rates = torch.exp(-diff.pow(2) / (2 * self.sigma ** 2))
        if not self.temporal_output:
            spikes_flat = (torch.rand(*rates.shape, device=x.device) < rates).float()
            return spikes_flat.reshape(*shape[:-1], shape[-1] * self.N)
        out = torch.stack([
            (torch.rand(*rates.shape, device=x.device) < rates).float().reshape(*shape[:-1], shape[-1] * self.N)
            for _ in range(self.T)
        ])
        return out
