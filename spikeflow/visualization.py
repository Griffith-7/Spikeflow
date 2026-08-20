"""Spike visualization tools for debugging and analysis."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn

if TYPE_CHECKING:
    import matplotlib.pyplot as plt


class SpikeMonitor:
    """Monitor that records spike trains and membrane potentials.

    Usage:
        monitor = SpikeMonitor(model)
        for t in range(T):
            out = model(x)
            monitor.record()
        monitor.plot_raster()  # spike raster plot
        monitor.plot_membrane()  # membrane potential traces
    """

    def __init__(self, module: nn.Module):
        self.module = module
        self.spikes: list[torch.Tensor] = []
        self.membranes: list[torch.Tensor] = []

    def record(self):
        """Record current spike/membrane state from all neurons."""
        for m in self.module.modules():
            if hasattr(m, "v") and m.v is not None:
                self.membranes.append(m.v.detach().cpu())
            if hasattr(m, "threshold_module"):
                # Record the pre-threshold voltage (what gets thresholded)
                if hasattr(m, "v") and m.v is not None:
                    self.spikes.append(m.v.detach().cpu())

    def reset(self):
        self.spikes.clear()
        self.membranes.clear()

    def plot_raster(self, ax: plt.Axes | None = None):
        """Plot spike raster: all neurons' spike times across timesteps."""
        import matplotlib.pyplot as plt

        if not self.spikes:
            raise ValueError("No spikes recorded. Call record() first.")
        ax = ax or plt.gca()
        spikes = torch.stack(self.spikes)
        T = spikes.shape[0]
        for t in range(T):
            fired = (spikes[t] > 0).flatten()
            indices = fired.nonzero(as_tuple=True)[0]
            ax.scatter([t] * len(indices), indices, s=1, c="black")
        ax.set_xlabel("Timestep")
        ax.set_ylabel("Neuron index")
        ax.set_title("Spike Raster Plot")
        return ax

    def plot_membrane(self, neuron_idx: int = 0, ax: plt.Axes | None = None):
        """Plot membrane potential trace for a specific neuron."""
        import matplotlib.pyplot as plt

        if not self.membranes:
            raise ValueError("No membrane potentials recorded. Call record() first.")
        ax = ax or plt.gca()
        membranes = torch.stack(self.membranes)
        trace = membranes[:, neuron_idx].flatten()
        ax.plot(range(len(trace)), trace)
        ax.set_xlabel("Timestep")
        ax.set_ylabel("Membrane potential")
        ax.set_title(f"Membrane Potential (neuron {neuron_idx})")
        return ax

    def get_spike_rate(self) -> float:
        """Get average firing rate across all recorded timesteps."""
        if not self.spikes:
            return 0.0
        spikes = torch.stack(self.spikes)
        return (spikes > 0).float().mean().item()
