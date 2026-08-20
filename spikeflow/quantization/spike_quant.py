"""Spike-based weight quantization for extreme compression.

This is the key to fitting 1B params on 4GB RAM:
- Binary weights (1-bit): 1B params = 125MB
- INT4 weights: 1B params = 500MB
- Combined with spike encoding: massive memory savings
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class BinaryWeightQuantizer(nn.Module):
    """Quantizes weights to {-1, +1} using sign function.

    At inference, the binary weight matrix can be stored as a bitpack,
    achieving 32x compression over FP32.

    During training, uses straight-through estimator for gradients.
    """

    def __init__(self, weight: Tensor | None = None):
        super().__init__()

    def forward(self, weight: Tensor) -> Tensor:
        if self.training:
            # Straight-through estimator
            return weight.sign() + weight - weight.detach()
        return weight.sign()

    @staticmethod
    def pack_binary(weights: Tensor) -> Tensor:
        """Pack binary weights into bits for 32x compression."""
        binary = (weights.sign() > 0).to(torch.uint8)
        # Reshape to pack 8 weights per byte
        packed = binary.view(-1, 8)
        # Convert each group of 8 bits to a byte
        bits = 2 ** torch.arange(8, device=weights.device, dtype=torch.uint8)
        packed = (packed * bits.unsqueeze(0)).sum(dim=-1)
        return packed

    @staticmethod
    def unpack_binary(packed: Tensor, shape: tuple[int, ...]) -> Tensor:
        """Unpack binary weights back to full shape."""
        bits = 2 ** torch.arange(8, device=packed.device, dtype=torch.uint8)
        unpacked = ((packed.unsqueeze(-1) >> bits.unsqueeze(0)) & 1).to(torch.float32)
        unpacked = unpacked.view(shape)
        return unpacked * 2 - 1  # Convert {0,1} to {-1, +1}


class SpikeQuantizer(nn.Module):
    """INT8/INT4 weight quantization optimized for spike-based inference.

    Quantization-aware training (QAT) for deployment on edge devices.
    """

    def __init__(self, bits: int = 8, symmetric: bool = True):
        super().__init__()
        self.bits = bits
        self.symmetric = symmetric
        self.qmax = 2 ** (bits - 1) - 1 if symmetric else 2**bits - 1

    def forward(self, weight: Tensor) -> Tensor:
        if self.training:
            # Fake quantization with straight-through estimator
            scale = weight.abs().max() / self.qmax
            if scale == 0:
                return weight
            wq = torch.clamp(weight / scale, -self.qmax, self.qmax)
            wq = torch.round(wq)
            wq = wq * scale
            return weight + (wq - weight).detach()
        else:
            scale = weight.abs().max() / self.qmax
            if scale == 0:
                return weight
            wq = torch.clamp(weight / scale, -self.qmax, self.qmax)
            wq = torch.round(wq).to(torch.int8 if self.bits == 8 else torch.int32)
            return wq, scale

    def extra_repr(self) -> str:
        return f"bits={self.bits}, symmetric={self.symmetric}"
