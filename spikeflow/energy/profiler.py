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

from spikeflow.neurons.base import BaseNeuron


@dataclass
class EnergyProfile:
    """Energy profile for a model."""
    model_name: str
    total_synops: int = 0
    total_macs: int = 0
    total_params: int = 0
    energy_synops_mj: float = 0.0
    energy_macs_mj: float = 0.0
    energy_mem_mj: float = 0.0
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
        """Profile an SNN model's energy consumption.

        Runs the model for ``timesteps`` steps with hooks attached:
          - compute layers record real output shapes so per-layer op counts
            include spatial positions and token counts;
          - spiking neurons record their actual output spike fraction, which
            scales SynOps (only fired spikes drive downstream computation).
        """
        model.eval()
        params = list(model.parameters())
        device = params[0].device if params else torch.device("cpu")

        profile = EnergyProfile(model_name=type(model).__name__)
        profile.total_params = sum(p.numel() for p in model.parameters())

        # --- hooks: capture layer output shapes and internal spike rates ---
        layer_shapes: dict[int, tuple] = {}
        spike_events = [0.0]
        spike_elements = [0]

        def shape_hook(idx):
            def hook(_m, _inp, out):
                out_shape = out.shape if isinstance(out, torch.Tensor) else out[0].shape
                layer_shapes[idx] = tuple(out_shape)
            return hook

        compute_modules = {}
        idx = 0
        for m in model.modules():
            if isinstance(m, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                compute_modules[idx] = m
                m.register_forward_hook(shape_hook(idx))
                idx += 1

        def neuron_hook(_m, _inp, out):
            out_t = out if isinstance(out, torch.Tensor) else out[0]
            spike_events[0] += (out_t > 0).sum().item()
            spike_elements[0] += out_t.numel()

        neuron_handles = [
            m.register_forward_hook(neuron_hook)
            for m in model.modules()
            if isinstance(m, BaseNeuron)
        ]

        x = torch.randn(*input_shape, device=device)
        for m in model.modules():
            if hasattr(m, "reset_state"):
                m.reset_state()

        try:
            for _ in range(timesteps):
                model(x)
        finally:
            for h in neuron_handles:
                h.remove()

        internal_spike_rate = (
            spike_events[0] / spike_elements[0] if spike_elements[0] else 0.0
        )
        profile.spike_rate = internal_spike_rate

        ops_per_step = self._count_ops(compute_modules, layer_shapes, input_shape)

        # SynOps: every timestep, each synapse processes an event only when
        # the upstream neuron actually spiked -> scale by measured rate.
        profile.total_synops = int(ops_per_step * timesteps * internal_spike_rate)
        # ANN-equivalent MACs: one non-spiking pass over the same graph.
        profile.total_macs = ops_per_step

        # Compute energy. Compute energies (SynOps vs MACs) are compared
        # directly; weight-memory traffic is reported separately because on
        # GPUs it dominates and would mask the synaptic-operation saving.
        synop_energy = profile.total_synops * self.hw["add_energy_pj"] / 1e9  # to mJ
        mac_energy = profile.total_macs * self.hw["mac_energy_pj"] / 1e9
        mem_energy = profile.total_params * 4 * timesteps * self.hw["mem_energy_pj"] / 1e9  # FP32

        profile.energy_synops_mj = synop_energy
        profile.energy_macs_mj = mac_energy
        profile.energy_mem_mj = mem_energy
        profile.energy_saving_pct = (1 - profile.energy_synops_mj / max(profile.energy_macs_mj, 1e-10)) * 100

        return profile

    @staticmethod
    def _layer_ops(module: nn.Module, out_shape: tuple | None, input_shape: tuple) -> int:
        """Op count for one compute layer for a single timestep, using the
        captured output shape when available."""
        if isinstance(module, nn.Linear):
            tokens = 1
            for s in out_shape[1:-1] if out_shape is not None and len(out_shape) > 2 else []:
                tokens *= s
            return module.in_features * module.out_features * tokens
        if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            k = 1
            for s in module.kernel_size:
                k *= s
            cin_eff = module.in_channels // module.groups
            spatial = None
            if out_shape is not None:
                spatial = 1
                for s in out_shape[2:]:
                    spatial *= s
            else:
                spatial = 1
                for s in input_shape[2:]:
                    spatial *= s
            return module.out_channels * cin_eff * k * spatial
        return 0

    def _count_ops(self, compute_modules: dict, layer_shapes: dict, input_shape: tuple) -> int:
        """Total ops across all compute layers for one timestep."""
        total = 0
        for idx, module in compute_modules.items():
            total += self._layer_ops(module, layer_shapes.get(idx), input_shape)
        return max(total, 1)

    def estimate_ann_macs(self, model: nn.Module, input_shape: tuple[int, ...]) -> int:
        """Estimate single-pass MAC operations for the equivalent ANN graph."""
        compute_modules = {
            i: m
            for i, m in enumerate(
                mod for mod in model.modules()
                if isinstance(mod, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d))
            )
        }
        return self._count_ops(compute_modules, {}, input_shape)

    def compare(self, snn_profile: EnergyProfile, ann_macs: int) -> dict[str, Any]:
        """Compare SNN vs ANN compute energy (memory traffic excluded)."""
        ann_energy = ann_macs * self.hw["mac_energy_pj"] / 1e9
        return {
            "snn_energy_mj": snn_profile.energy_synops_mj,
            "ann_energy_mj": ann_energy,
            "snn_memory_energy_mj": snn_profile.energy_mem_mj,
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
            f"  Energy (SNN compute): {profile.energy_synops_mj:>10.3f} mJ",
            f"  Energy (ANN compute): {profile.energy_macs_mj:>10.3f} mJ",
            f"  Energy (weight memory): {profile.energy_mem_mj:>8.3f} mJ",
            f"  Energy Saving:    {profile.energy_saving_pct:>11.1f}%",
            "=" * 60,
        ]
        return "\n".join(lines)

    def _estimate_macs(self, model: nn.Module, input_shape: tuple) -> int:
        """Backward-compatible alias for :meth:`estimate_ann_macs`."""
        return self.estimate_ann_macs(model, input_shape)
