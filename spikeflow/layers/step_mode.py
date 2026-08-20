"""Step-mode module wrapper for single-step (s) and multi-step (m) computation.

Allows switching between sequential (debuggable) and parallel (fast) modes.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class StepModeModule(nn.Module):
    """Wrapper that enables single-step or multi-step computation.

    In 's' mode: forward() processes one timestep, caller manages the loop.
    In 'm' mode: forward() processes all T timesteps internally.

    Usage:
        model = StepModeModule(SpikingResNet18(...), mode='s')
        # Single-step: caller loops
        for t in range(T):
            out = model(x)

        model.set_mode('m')
        # Multi-step: model loops internally
        out = model(x, T=4)
    """

    def __init__(self, module: nn.Module, mode: str = "s"):
        super().__init__()
        self.module = module
        self._mode = mode

    def set_mode(self, mode: str):
        assert mode in ("s", "m"), f"Mode must be 's' or 'm', got '{mode}'"
        self._mode = mode

    def forward(self, x: torch.Tensor, timesteps: int | None = None) -> torch.Tensor:
        if self._mode == "s" or timesteps is None or timesteps == 1:
            return self.module(x)

        self._reset_states()
        acc = None
        for _ in range(timesteps):
            out = self.module(x)
            acc = out if acc is None else acc + out
        return acc / timesteps

    def _reset_states(self):
        for m in self.module.modules():
            if hasattr(m, "reset_state"):
                m.reset_state()

    def reset_state(self):
        self._reset_states()

    def set_sfa_mode(self, enabled: bool):
        for m in self.module.modules():
            if hasattr(m, "set_sfa_mode"):
                m.set_sfa_mode(enabled)

    def set_readout(self, enabled: bool):
        for m in self.module.modules():
            if hasattr(m, "set_readout"):
                m.set_readout(enabled)
