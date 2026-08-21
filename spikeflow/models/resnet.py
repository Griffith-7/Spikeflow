"""Spiking ResNet — ImageNet-scale CNN with spiking neurons.

Replaces ReLU with LIF neurons in standard ResNet architecture.
Same skip connections, same depth, but spike-based computation.

Accuracy targets:
    SpikingResNet18: ~70% ImageNet (vs 69.8% ResNet18 ANN)
    SpikingResNet34: ~74% ImageNet (vs 73.3% ResNet34 ANN)
    SpikingResNet50: ~76% ImageNet (vs 76.1% ResNet50 ANN)
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from spikeflow.layers.linear import SpikingLinear
from spikeflow.neurons.lif import LIFNode


class SpikingBasicBlock(nn.Module):
    """Basic residual block with spiking neurons (ResNet-18/34).

    Ordering is Conv -> BN -> LIF: BatchNorm always sees conv outputs, so its
    running statistics stay consistent between SFA training (neuron = ReLU)
    and T=D spike inference (neuron emits binary spikes). The residual sum is
    passed through the block's output neuron.
    """

    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None, threshold=1.0, tau=2.0):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.neuron1 = LIFNode(threshold=threshold, tau=tau)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.neuron2 = LIFNode(threshold=threshold, tau=tau)
        self.downsample = downsample

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = self.neuron1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out = out + identity
        return self.neuron2(out)


class SpikingBottleneck(nn.Module):
    """Bottleneck residual block with spiking neurons (ResNet-50/101/152).

    Same Conv -> BN -> LIF ordering as :class:`SpikingBasicBlock`.
    """

    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None, threshold=1.0, tau=2.0):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.neuron1 = LIFNode(threshold=threshold, tau=tau)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.neuron2 = LIFNode(threshold=threshold, tau=tau)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.neuron3 = LIFNode(threshold=threshold, tau=tau)
        self.downsample = downsample

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = self.neuron1(self.bn1(self.conv1(x)))
        out = self.neuron2(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out = out + identity
        return self.neuron3(out)


class SpikingResNet(nn.Module):
    """Spiking ResNet for image classification.

    Drop-in replacement for torchvision's ResNet with spiking neurons.
    """

    def __init__(
        self,
        block,
        layers: list[int],
        num_classes: int = 1000,
        in_channels: int = 3,
        threshold: float = 1.0,
        tau: float = 2.0,
    ):
        super().__init__()

        self.in_channels = 64
        # Stem: Conv -> BN -> LIF -> MaxPool
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.neuron1 = LIFNode(threshold=threshold, tau=tau)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0], threshold=threshold, tau=tau)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, threshold=threshold, tau=tau)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, threshold=threshold, tau=tau)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, threshold=threshold, tau=tau)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = SpikingLinear(512 * block.expansion, num_classes, threshold=threshold, tau=tau, readout=True)

        self._init_weights()

    def _make_layer(self, block, out_channels, blocks, stride=1, threshold=1.0, tau=2.0):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            # Shortcut: Conv -> BN (no neuron) so the residual add happens on
            # analog magnitudes before the block's output neuron.
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )
        layers = [block(self.in_channels, out_channels, stride, downsample, threshold=threshold, tau=tau)]
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels, threshold=threshold, tau=tau))
        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: Tensor) -> Tensor:
        x = self.pool(self.neuron1(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)

    def reset_state(self):
        for m in self.modules():
            if m is not self and hasattr(m, "reset_state"):
                m.reset_state()

    def set_sfa_mode(self, enabled: bool):
        for m in self.modules():
            if m is not self and hasattr(m, "set_sfa_mode"):
                m.set_sfa_mode(enabled)

    def set_readout(self, enabled: bool):
        for m in self.modules():
            if m is not self and hasattr(m, "set_readout"):
                m.set_readout(enabled)


def SpikingResNet18(num_classes=1000, **kwargs):
    return SpikingResNet(SpikingBasicBlock, [2, 2, 2, 2], num_classes=num_classes, **kwargs)

def SpikingResNet34(num_classes=1000, **kwargs):
    return SpikingResNet(SpikingBasicBlock, [3, 4, 6, 3], num_classes=num_classes, **kwargs)

def SpikingResNet50(num_classes=1000, **kwargs):
    return SpikingResNet(SpikingBottleneck, [3, 4, 6, 3], num_classes=num_classes, **kwargs)
