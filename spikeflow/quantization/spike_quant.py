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
        """Pack binary weights into bits for 32x compression.

        Element j of each group of 8 is stored at bit position j.
        The tensor is zero-padded up to a multiple of 8, so any shape works.
        """
        binary = (weights.sign() > 0).to(torch.uint8).reshape(-1)
        pad = (-binary.numel()) % 8
        if pad:
            binary = torch.cat(
                [binary, torch.zeros(pad, dtype=torch.uint8, device=binary.device)]
            )
        # Bit *positions* 0..7 (not values 2**i) — positions are what the
        # shift operator expects on both pack and unpack.
        bits = torch.arange(8, device=weights.device, dtype=torch.uint8)
        return (binary.view(-1, 8) << bits).sum(dim=-1)

    @staticmethod
    def unpack_binary(packed: Tensor, shape: tuple[int, ...]) -> Tensor:
        """Unpack binary weights back to full shape (inverse of pack_binary)."""
        total = 1
        for s in shape:
            total *= s
        bits = torch.arange(8, device=packed.device, dtype=torch.uint8)
        unpacked = ((packed.reshape(-1, 1) >> bits) & 1).to(torch.float32).reshape(-1)
        if unpacked.numel() < total:
            raise ValueError(
                f"packed data holds {unpacked.numel()} weights, need {total} for shape {shape}"
            )
        return (unpacked[:total] * 2 - 1).reshape(shape)


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
        """Fake quantization (straight-through estimator).

        Returns a tensor with the same shape and dtype as ``weight`` in both
        train and eval mode. Use :meth:`quantize` to obtain the integer
        payload plus scale for deployment.
        """
        scale = weight.abs().max() / self.qmax
        if scale == 0:
            return weight
        wq = torch.clamp(weight / scale, -self.qmax, self.qmax)
        wq = torch.round(wq) * scale
        if self.training:
            return weight + (wq - weight).detach()
        return wq

    def quantize(self, weight: Tensor) -> tuple[Tensor, Tensor]:
        """Produce the deployable (integer weights, scale) pair."""
        scale = weight.abs().max() / self.qmax
        if scale == 0:
            raise ValueError("Cannot quantize an all-zero tensor (scale is 0)")
        wq = torch.clamp(weight / scale, -self.qmax, self.qmax).round()
        dtype = torch.int8 if self.bits == 8 else torch.int32
        return wq.to(dtype), scale

    def extra_repr(self) -> str:
        return f"bits={self.bits}, symmetric={self.symmetric}"
