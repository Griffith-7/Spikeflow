"""Spike Firing Approximation (SFA) training engine.

The core innovation: train at T=1 with standard backprop (like transformers),
then deploy at T=D with full spike dynamics. This eliminates the 4-10x
training overhead of traditional SNN training via BPTT.
"""

from __future__ import annotations

import warnings
from typing import Any

import torch
import torch.nn as nn


def _set_sfa_recursive(module: nn.Module, enabled: bool):
    """Recursively enable/disable SFA mode on all SpikeFlow modules."""
    for child in module.modules():
        if hasattr(child, "set_sfa_mode"):
            child.set_sfa_mode(enabled)


class EMA:
    """Exponential Moving Average of model weights for better generalization."""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.model = model
        self.decay = decay
        self.shadow = {name: param.clone().detach() for name, param in model.named_parameters()}

    @torch.no_grad()
    def update(self):
        for name, param in self.model.named_parameters():
            self.shadow[name].mul_(self.decay).add_(param, alpha=1 - self.decay)

    def apply_shadow(self):
        if getattr(self, "_applied", False):
            raise RuntimeError(
                "EMA.apply_shadow() called twice without restore(). "
                "Call restore() first or the backup would be corrupted."
            )
        self._backup = {name: param.clone() for name, param in self.model.named_parameters()}
        for name, param in self.model.named_parameters():
            param.data.copy_(self.shadow[name])
        self._applied = True

    def restore(self):
        if not getattr(self, "_applied", False):
            return
        for name, param in self.model.named_parameters():
            param.data.copy_(self._backup[name])
        self._applied = False


class SFATrainer:
    """Trainer that implements Spike Firing Approximation.

    SFA Training Protocol:
        1. Enable SFA mode: all neurons behave as ReLU
        2. Train with standard backprop at T=1 (same speed as transformers)
        3. Disable SFA: neurons use real spike dynamics
        4. Inference at T=D timesteps

    This means training cost = transformer training cost.
    Only inference adds the temporal dimension.

    Usage:
        model = SpikingTransformer(...)
        trainer = SFATrainer(model, optimizer, scheduler)

        # Training loop — runs at T=1 speed
        for epoch in range(epochs):
            trainer.train_sfa(train_loader)

        # Switch to spike mode for inference
        trainer.enable_spike_mode(timesteps=4)
        results = trainer.evaluate(test_loader)
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any = None,
        device: torch.device | str = "cuda",
        grad_clip: float = 1.0,
        use_mixed_precision: bool = True,
        ema_decay: float = 0.0,
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = torch.device(device) if isinstance(device, str) else device
        self.grad_clip = grad_clip

        # AMP is only reliable on CUDA; silently running autocast("cuda") on a
        # CPU device breaks training, so disable it with a warning instead.
        if use_mixed_precision and self.device.type != "cuda":
            warnings.warn(
                f"use_mixed_precision=True requires a CUDA device, got "
                f"'{self.device.type}'; disabling mixed precision.",
                stacklevel=2,
            )
            use_mixed_precision = False
        self.use_amp = use_mixed_precision
        self.scaler = torch.amp.GradScaler(self.device.type) if self.use_amp else None
        self.ema = EMA(model, decay=ema_decay) if ema_decay > 0 else None

    def enable_sfa_mode(self):
        """Switch model to SFA mode (ReLU-like, T=1 training).

        Readout configuration is intentionally left untouched: it is a static
        per-layer property (classification heads set ``readout=True`` at
        construction). Toggling it globally would either break the head or —
        worse — turn every hidden neuron into an analog readout during spike
        inference, eliminating all binary spikes.
        """
        _set_sfa_recursive(self.model, enabled=True)

    def enable_spike_mode(self, timesteps: int = 4):
        """Switch model to spike mode for inference.

        Only SFA mode is disabled; neurons configured with ``readout=True``
        (typically the classification head) keep returning membrane
        potentials while all hidden neurons emit binary spikes.
        """
        _set_sfa_recursive(self.model, enabled=False)
        self.timesteps = timesteps

    def train_sfa(
        self,
        train_loader: torch.utils.data.DataLoader,
        criterion: nn.Module | None = None,
        epoch: int = 0,
    ) -> dict[str, float]:
        """Train one epoch in SFA mode (T=1, like transformer training).

        Returns dict with 'loss', 'accuracy', etc.
        """
        self.enable_sfa_mode()
        self.model.train()

        if criterion is None:
            criterion = nn.CrossEntropyLoss()

        total_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (data, targets) in enumerate(train_loader):
            data, targets = data.to(self.device), targets.to(self.device)

            self.optimizer.zero_grad()

            # Forward pass — single step, like transformer
            if self.scaler:
                with torch.amp.autocast(self.device.type):
                    output = self.model(data)
                    loss = criterion(output, targets)
                self.scaler.scale(loss).backward()
                if self.grad_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                output = self.model(data)
                loss = criterion(output, targets)
                loss.backward()
                if self.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optimizer.step()

            if self.scheduler and isinstance(self.scheduler, torch.optim.lr_scheduler.OneCycleLR):
                self.scheduler.step()

            total_loss += loss.item() * data.size(0)
            pred = output.argmax(dim=-1)
            correct += pred.eq(targets).sum().item()
            total += data.size(0)

        if self.scheduler and not isinstance(self.scheduler, torch.optim.lr_scheduler.OneCycleLR):
            self.scheduler.step()

        if self.ema:
            self.ema.update()

        return {
            "loss": total_loss / total,
            "accuracy": correct / total,
        }

    @torch.no_grad()
    def evaluate(
        self,
        data_loader: torch.utils.data.DataLoader,
        criterion: nn.Module | None = None,
        timesteps: int | None = None,
        use_ema: bool = True,
    ) -> dict[str, float]:
        """Evaluate with full spike dynamics over T timesteps."""
        self.model.eval()
        if self.ema and use_ema:
            self.ema.apply_shadow()
        try:
            T = timesteps or getattr(self, "timesteps", 4)

            if criterion is None:
                criterion = nn.CrossEntropyLoss()

            total_loss = 0.0
            correct = 0
            total = 0

            for data, targets in data_loader:
                data, targets = data.to(self.device), targets.to(self.device)
                self._reset_states()
                output_acc = None
                for t in range(T):
                    output = self.model(data)
                    output_acc = output if output_acc is None else output_acc + output
                loss = criterion(output_acc / T, targets)
                total_loss += loss.item() * data.size(0)
                pred = output_acc.argmax(dim=-1)
                correct += pred.eq(targets).sum().item()
                total += data.size(0)

            return {"loss": total_loss / total, "accuracy": correct / total}
        finally:
            if self.ema and use_ema:
                self.ema.restore()

    def _reset_states(self):
        """Reset all spiking neuron membrane potentials."""
        for module in self.model.modules():
            if hasattr(module, "reset_state"):
                module.reset_state()
