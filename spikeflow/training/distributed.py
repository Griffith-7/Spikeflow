"""Distributed training utilities for SNNs.

Supports:
- FSDP (Fully Sharded Data Parallel) for large models
- DataParallel for multi-GPU
- Gradient accumulation
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.utils.data import DataLoader, DistributedSampler


def setup_distributed(rank: int = 0, world_size: int = 1):
    """Initialize distributed training."""
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "12355")
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup_distributed():
    """Clean up distributed training."""
    dist.destroy_process_group()


def wrap_ddp(model: nn.Module, rank: int):
    """Wrap model with DistributedDataParallel."""
    from torch.nn.parallel import DistributedDataParallel
    model = model.to(rank)
    return DistributedDataParallel(model, device_ids=[rank])


def create_distributed_loader(
    dataset,
    batch_size: int,
    rank: int,
    world_size: int,
    num_workers: int = 4,
) -> DataLoader:
    """Create a distributed DataLoader with proper sampling."""
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )


class GradientAccumulator:
    """Accumulate gradients over multiple micro-batches for effective larger batch sizes."""

    def __init__(self, model: nn.Module, optimizer: torch.optim.Optimizer, accumulation_steps: int = 4):
        self.model = model
        self.optimizer = optimizer
        self.accumulation_steps = accumulation_steps
        self._step = 0

    def accumulate(self, loss: torch.Tensor):
        """Accumulate gradient from one micro-batch."""
        (loss / self.accumulation_steps).backward()
        self._step += 1
        if self._step % self.accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)

    def sync(self):
        """Force gradient sync (for logging, etc.)."""
        if self._step % self.accumulation_steps != 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
            self._step = 0


def get_fsdp_wrapping_policy():
    """Get FSDP wrapping policy for SpikeFlow models."""
    try:
        from torch.distributed.fsdp import MixedPrecision

        mp_policy = MixedPrecision(
            param_dtype=torch.float16,
            reduce_dtype=torch.float16,
            buffer_dtype=torch.float16,
        )
        return mp_policy
    except ImportError:
        return None
