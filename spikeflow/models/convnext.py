"""Spiking ConvNeXt — modern CNN architecture with spiking neurons.

ConvNeXt's large-kernel design is ideal for SNNs because:
1. Large kernels reduce the need for deep stacking
2. Depthwise convolution is cheaper in spike domain
3. LayerNorm + Channel interactions work well with spike rates
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from spikeflow.layers.linear import SpikingLinear
from spikeflow.neurons.lif import LIFNode


class SpikingConvNeXtBlock(nn.Module):
    """ConvNeXt block with spiking neurons.

    Architecture: DWConv -> LayerNorm -> Linear -> GELU -> Linear -> + residual
    """

    def __init__(self, dim: int, threshold: float = 1.0, tau: float = 2.0):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.neuron = LIFNode(threshold=threshold, tau=tau)
        self.pwconv2 = nn.Linear(4 * dim, dim)

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = torch.nn.functional.gelu(x)
        x = self.neuron(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)
        return residual + x

    def reset_state(self):
        self.neuron.reset_state()

    def set_sfa_mode(self, enabled: bool):
        self.neuron.set_sfa_mode(enabled)


class SpikingConvNeXtStage(nn.Module):
    """A stage of ConvNeXt blocks."""

    def __init__(self, dim: int, depth: int, threshold: float = 1.0, tau: float = 2.0):
        super().__init__()
        self.blocks = nn.ModuleList([
            SpikingConvNeXtBlock(dim, threshold=threshold, tau=tau)
            for _ in range(depth)
        ])
        # Downsampling layer (except last stage)
        self.downsample = None

    def forward(self, x: Tensor) -> Tensor:
        for block in self.blocks:
            x = block(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x

    def reset_state(self):
        for block in self.blocks:
            block.reset_state()

    def set_sfa_mode(self, enabled: bool):
        for block in self.blocks:
            block.set_sfa_mode(enabled)


class SpikingConvNeXt(nn.Module):
    """Spiking ConvNeXt for image classification.

    Drop-in replacement for torchvision/timm ConvNeXt.
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1000,
        dims: list[int] = [96, 192, 384, 768],
        depths: list[int] = [3, 3, 9, 3],
        threshold: float = 1.0,
        tau: float = 2.0,
    ):
        super().__init__()
        self.dims = dims
        self.depths = depths

        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, dims[0], kernel_size=4, stride=4),
        )

        # Stages
        self.stages = nn.ModuleList()
        for i in range(4):
            stage = SpikingConvNeXtStage(dims[i], depths[i], threshold=threshold, tau=tau)
            if i < 3:
                stage.downsample = nn.Sequential(
                    nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2),
                )
            self.stages.append(stage)

        self.norm = nn.LayerNorm(dims[-1])
        self.head = SpikingLinear(dims[-1], num_classes, threshold=threshold, tau=tau, readout=True)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: Tensor) -> Tensor:
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        x = x.mean(dim=[2, 3])
        x = self.norm(x)
        return self.head(x)

    def reset_state(self):
        for stage in self.stages:
            stage.reset_state()

    def set_sfa_mode(self, enabled: bool):
        for stage in self.stages:
            stage.set_sfa_mode(enabled)


def SpikingConvNeXtTiny(num_classes=1000, **kwargs):
    return SpikingConvNeXt(dims=[96, 192, 384, 768], depths=[3, 3, 9, 3], num_classes=num_classes, **kwargs)

def SpikingConvNeXtSmall(num_classes=1000, **kwargs):
    return SpikingConvNeXt(dims=[96, 192, 384, 768], depths=[3, 3, 27, 3], num_classes=num_classes, **kwargs)
