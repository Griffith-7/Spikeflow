"""Pre-Activation Spiking ResNet — better gradient flow for SNNs.

PreAct (BN→ReLU→Conv) outperforms post-activation (Conv→BN→ReLU)
for spiking networks because BN runs on clean activations before the neuron.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from spikeflow.layers.conv import SpikingConv2d
from spikeflow.layers.linear import SpikingLinear


class PreActBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_ch, out_ch, stride=1, downsample=None, threshold=1.0, tau=2.0):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.conv1 = SpikingConv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False, threshold=threshold, tau=tau)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.conv2 = SpikingConv2d(out_ch, out_ch, 3, padding=1, bias=False, threshold=threshold, tau=tau)
        self.downsample = downsample

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = self.conv1(torch.relu(self.bn1(x)))
        out = self.conv2(torch.relu(self.bn2(out)))
        if self.downsample is not None:
            identity = self.downsample(x)
        return out + identity


class PreActBottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_ch, out_ch, stride=1, downsample=None, threshold=1.0, tau=2.0):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.conv1 = SpikingConv2d(in_ch, out_ch, 1, bias=False, threshold=threshold, tau=tau)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.conv2 = SpikingConv2d(out_ch, out_ch, 3, stride=stride, padding=1, bias=False, threshold=threshold, tau=tau)
        self.bn3 = nn.BatchNorm2d(out_ch)
        self.conv3 = SpikingConv2d(out_ch, out_ch * self.expansion, 1, bias=False, threshold=threshold, tau=tau)
        self.downsample = downsample

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = self.conv1(torch.relu(self.bn1(x)))
        out = self.conv2(torch.relu(self.bn2(out)))
        out = self.conv3(torch.relu(self.bn3(out)))
        if self.downsample is not None:
            identity = self.downsample(x)
        return out + identity


class SpikingPreActResNet(nn.Module):
    """Pre-Activation Spiking ResNet for CIFAR-10/ImageNet."""

    def __init__(self, block, layers, num_classes=10, in_channels=3, threshold=1.0, tau=2.0):
        super().__init__()
        self.in_channels = 16
        self.conv1 = SpikingConv2d(in_channels, 16, 3, padding=1, bias=False, threshold=threshold, tau=tau)
        self.layer1 = self._make_layer(block, 16, layers[0], threshold=threshold, tau=tau)
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2, threshold=threshold, tau=tau)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2, threshold=threshold, tau=tau)
        self.bn = nn.BatchNorm2d(64)
        self.fc = SpikingLinear(64 * block.expansion, num_classes, threshold=threshold, tau=tau, readout=True)
        self._init_weights()

    def _make_layer(self, block, out_ch, blocks, stride=1, threshold=1.0, tau=2.0):
        downsample = None
        if stride != 1 or self.in_channels != out_ch * block.expansion:
            if stride != 1:
                downsample = nn.Sequential(
                    nn.AvgPool2d(2),
                    SpikingConv2d(self.in_channels, out_ch * block.expansion, 1, bias=False, threshold=threshold, tau=tau),
                )
            else:
                downsample = SpikingConv2d(self.in_channels, out_ch * block.expansion, 1, bias=False, threshold=threshold, tau=tau)
        layers = [block(self.in_channels, out_ch, stride, downsample, threshold=threshold, tau=tau)]
        self.in_channels = out_ch * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_ch, threshold=threshold, tau=tau))
        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, SpikingConv2d):
                nn.init.kaiming_normal_(m.conv.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: Tensor) -> Tensor:
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = torch.relu(self.bn(out))
        out = nn.functional.adaptive_avg_pool2d(out, 1).flatten(1)
        return self.fc(out)

    def reset_state(self):
        for m in self.children():
            if hasattr(m, "reset_state"):
                m.reset_state()

    def set_sfa_mode(self, enabled: bool):
        for m in self.children():
            if hasattr(m, "set_sfa_mode"):
                m.set_sfa_mode(enabled)

    def set_readout(self, enabled: bool):
        for m in self.children():
            if hasattr(m, "set_readout"):
                m.set_readout(enabled)


def SpikingPreActResNet20(num_classes=10, **kwargs):
    return SpikingPreActResNet(PreActBasicBlock, [3, 3, 3], num_classes=num_classes, **kwargs)


def SpikingPreActResNet56(num_classes=10, **kwargs):
    return SpikingPreActResNet(PreActBasicBlock, [9, 9, 9], num_classes=num_classes, **kwargs)


def SpikingPreActResNet110(num_classes=10, **kwargs):
    return SpikingPreActResNet(PreActBasicBlock, [18, 18, 18], num_classes=num_classes, **kwargs)
