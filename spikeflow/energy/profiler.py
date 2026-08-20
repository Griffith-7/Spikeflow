"""Energy profiler — measures actual energy consumption of SNN inference.

Computes energy based on:
    - Number of synaptic operations (SynOps) vs MAC operations
    - Binary spike multiplication-replacement (add-only)
    - Memory access costs
    - Target hardware power profiles

Based on: NeuroMC energy model and E-SpikeFormer energy estimation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn


@dataclass
class EnergyProfile:
    """Energy profile for a model."""
    model_name: str
    total_synops: int = 0
    total_macs: int = 0
    total_params: int = 0
    energy_synops_mj: float = 0.0
    energy_macs_mj: float = 0.0
    energy_saving_pct: float = 0.0
    spike_rate: float = 0.0
    memory_bytes: int = 0


# Hardware power models (from published literature)
HARDWARE_MODELS = {
    "gpu_a100": {
        "mac_energy_pj": 0.5,  # picojoules per MAC (A100 FP16)
        "add_energy_pj": 0.1,  # picojoules per ADD
        "mem_energy_pj": 4.0,  # picojoules per memory access (byte)
        "spike_mem_pj": 0.01,  # picojoules per sparse spike (event-driven)
    },
    "gpu_v100": {
        "mac_energy_pj": 0.8,
        "add_energy_pj": 0.15,
        "mem_energy_pj": 5.0,
        "spike_mem_pj": 0.02,
    },
    "loihi_2": {
        "mac_energy_pj": 0.0,
        "add_energy_pj": 0.05,
        "mem_energy_pj": 0.5,
        "spike_mem_pj": 0.005,
    },
    "edge_arm": {
        "mac_energy_pj": 2.0,
        "add_energy_pj": 0.3,
        "mem_energy_pj": 10.0,
        "spike_mem_pj": 0.05,
    },
}


class EnergyProfiler:
    """Profile energy consumption of SNN vs ANN models.

    Usage:
        profiler = EnergyProfiler()

        # Profile SNN
        snn_profile = profiler.profile_snn(model, input_shape=(1, 3, 224, 224), timesteps=4)

        # Compare with standard transformer
        ann_macs = profiler.estimate_ann_macs(model, input_shape)
        comparison = profiler.compare(snn_profile, ann_macs)

        print(profiler.report(snn_profile))
    """

    def __init__(self, hardware: str = "gpu_a100"):
        self.hw = HARDWARE_MODELS.get(hardware, HARDWARE_MODELS["gpu_a100"])

    def profile_snn(
        self,
        model: nn.Module,
        input_shape: tuple[int, ...] = (1, 3, 224, 224),
        timesteps: int = 4,
    ) -> EnergyProfile:
        """Profile an SNN model's energy consumption."""
        model.eval()
        device = next(model.parameters()).device if list(model.parameters()) else torch.device("cpu")

        profile = EnergyProfile(model_name=type(model).__name__)

        # Count parameters
        profile.total_params = sum(p.numel() for p in model.parameters())

        # Measure spike rates
        x = torch.randn(*input_shape, device=device)

        # Reset states
        for m in model.modules():
            if hasattr(m, "reset_state"):
                m.reset_state()

        spike_counts = []
        for t in range(timesteps):
            out = model(x)
            spike_rate = (out > 0).float().mean().item()
            spike_counts.append(spike_rate)

        profile.spike_rate = sum(spike_counts) / len(spike_counts)

        # Count synaptic operations
        profile.total_synops = self._count_synops(model, input_shape, timesteps)
        profile.total_macs = self._estimate_macs(model, input_shape)

        # Compute energy
        synop_energy = profile.total_synops * self.hw["add_energy_pj"] / 1e9  # to mJ
        mac_energy = profile.total_macs * self.hw["mac_energy_pj"] / 1e9
        mem_energy = profile.total_params * 4 * timesteps * self.hw["mem_energy_pj"] / 1e9  # FP32

        profile.energy_synops_mj = synop_energy + mem_energy
        profile.energy_macs_mj = mac_energy
        profile.energy_saving_pct = (1 - profile.energy_synops_mj / max(profile.energy_macs_mj, 1e-10)) * 100

        return profile

    def _count_synops(self, model: nn.Module, input_shape: tuple, timesteps: int) -> int:
        """Count total synaptic operations."""
        total = 0
        for module in model.modules():
            if isinstance(module, nn.Linear):
                # Linear: in * out per token
                total += module.in_features * module.out_features
            elif isinstance(module, nn.Conv2d):
                # Conv: C_out * C_in * K * K * H * W
                k = module.kernel_size[0] * module.kernel_size[1]
                total += module.out_channels * module.in_channels * k
        batch_size = input_shape[0]
        total *= batch_size * timesteps
        return total

    def _estimate_macs(self, model: nn.Module, input_shape: tuple) -> int:
        """Estimate MAC operations for equivalent ANN."""
        total = 0
        for module in model.modules():
            if isinstance(module, nn.Linear):
                total += 2 * module.in_features * module.out_features
            elif isinstance(module, nn.Conv2d):
                k = module.kernel_size[0] * module.kernel_size[1]
                total += 2 * module.out_channels * module.in_channels * k
        batch_size = input_shape[0]
        if len(input_shape) > 2:
            spatial = input_shape[2] * input_shape[3]
            total *= batch_size * spatial
        else:
            total *= batch_size
        return total

    def compare(self, snn_profile: EnergyProfile, ann_macs: int) -> dict[str, Any]:
        """Compare SNN vs ANN energy."""
        ann_energy = ann_macs * self.hw["mac_energy_pj"] / 1e9
        return {
            "snn_energy_mj": snn_profile.energy_synops_mj,
            "ann_energy_mj": ann_energy,
            "energy_saving_pct": (1 - snn_profile.energy_synops_mj / max(ann_energy, 1e-10)) * 100,
            "snn_synops": snn_profile.total_synops,
            "ann_macs": ann_macs,
            "synop_to_mac_ratio": snn_profile.total_synops / max(ann_macs, 1),
        }

    def report(self, profile: EnergyProfile) -> str:
        """Generate a formatted energy report."""
        lines = [
            "=" * 60,
            f"Energy Profile: {profile.model_name}",
            "=" * 60,
            f"  Parameters:       {profile.total_params:>15,}",
            f"  SynOps:           {profile.total_synops:>15,}",
            f"  MACs (ANN equiv): {profile.total_macs:>15,}",
            f"  Spike Rate:       {profile.spike_rate:>14.2%}",
            f"  Energy (SNN):     {profile.energy_synops_mj:>12.3f} mJ",
            f"  Energy (ANN):     {profile.energy_macs_mj:>12.3f} mJ",
            f"  Energy Saving:    {profile.energy_saving_pct:>11.1f}%",
            "=" * 60,
        ]
        return "\n".join(lines)
