"""Mixed precision training utilities for SNNs.

Supports FP16 and BF16 training with automatic loss scaling.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


class MixedPrecisionSFA:
    """SFA trainer with built-in mixed precision support.

    Automatically handles:
    - FP16/BF16 forward pass
    - Gradient scaling
    - Master weight updates
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any = None,
        device: torch.device | str = "cuda",
        dtype: torch.dtype = torch.float16,
        grad_clip: float = 1.0,
        ema_decay: float = 0.0,
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = torch.device(device) if isinstance(device, str) else device
        self.dtype = dtype
        self.grad_clip = grad_clip
        self.scaler = torch.amp.GradScaler(enabled=(dtype == torch.float16))

        from spikeflow.training.sfa import EMA
        self.ema = EMA(model, decay=ema_decay) if ema_decay > 0 else None

    def train_step(self, x: torch.Tensor, y: torch.Tensor, criterion: nn.Module) -> dict[str, float]:
        """Single training step with mixed precision."""
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        with torch.autocast("cuda", dtype=self.dtype):
            output = self.model(x)
            loss = criterion(output, y)

        self.scaler.scale(loss).backward()
        if self.grad_clip > 0:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        self.scaler.step(self.optimizer)
        self.scaler.update()

        if self.ema:
            self.ema.update()

        return {"loss": loss.item(), "accuracy": output.argmax(-1).eq(y).float().mean().item()}

    @torch.no_grad()
    def evaluate(self, x: torch.Tensor, y: torch.Tensor, criterion: nn.Module, timesteps: int = 4) -> dict[str, float]:
        """Evaluate with mixed precision and T timesteps."""
        self.model.eval()
        if self.ema:
            self.ema.apply_shadow()
        try:
            for m in self.model.modules():
                if hasattr(m, "reset_state"):
                    m.reset_state()
            acc = None
            for _ in range(timesteps):
                with torch.autocast("cuda", dtype=self.dtype):
                    out = self.model(x)
                acc = out if acc is None else acc + out
            loss = criterion(acc / timesteps, y)
            return {"loss": loss.item(), "accuracy": acc.argmax(-1).eq(y).float().mean().item()}
        finally:
            if self.ema:
                self.ema.restore()
