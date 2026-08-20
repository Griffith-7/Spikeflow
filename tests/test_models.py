# tests/test_models.py
"""Tests for complete model zoo."""

import torch

from spikeflow.models.convnext import SpikingConvNeXtTiny
from spikeflow.models.resnet import SpikingResNet18, SpikingResNet50
from spikeflow.models.vit import SpikingViTSmall, SpikingViTTiny


class TestSpikingViT:
    def test_tiny_forward(self):
        model = SpikingViTTiny(num_classes=10)
        x = torch.randn(2, 3, 224, 224)
        out = model(x)
        assert out.shape == (2, 10)

    def test_sfa_mode(self):
        model = SpikingViTTiny(num_classes=10)
        model.set_sfa_mode(True)
        x = torch.randn(2, 3, 224, 224)
        out = model(x)
        assert out.shape == (2, 10)
        # SFA output should not be binary
        assert not torch.all((out == 0) | (out == 1))

    def test_spike_mode(self):
        model = SpikingViTTiny(num_classes=10)
        model.set_sfa_mode(False)
        model.reset_state()
        x = torch.randn(2, 3, 224, 224)
        out = model(x)
        assert out.shape == (2, 10)

    def test_small_forward(self):
        model = SpikingViTSmall(num_classes=100)
        x = torch.randn(2, 3, 224, 224)
        out = model(x)
        assert out.shape == (2, 100)


class TestSpikingResNet:
    def test_resnet18_forward(self):
        model = SpikingResNet18(num_classes=10)
        x = torch.randn(2, 3, 32, 32)
        out = model(x)
        assert out.shape == (2, 10)

    def test_resnet50_forward(self):
        model = SpikingResNet50(num_classes=1000)
        x = torch.randn(2, 3, 224, 224)
        out = model(x)
        assert out.shape == (2, 1000)

    def test_sfa_mode(self):
        model = SpikingResNet18(num_classes=10)
        model.set_sfa_mode(True)
        x = torch.randn(2, 3, 32, 32)
        out = model(x)
        assert out.shape == (2, 10)


class TestSpikingConvNeXt:
    def test_tiny_forward(self):
        model = SpikingConvNeXtTiny(num_classes=10)
        x = torch.randn(2, 3, 224, 224)
        out = model(x)
        assert out.shape == (2, 10)
