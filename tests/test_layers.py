# tests/test_layers.py
"""Tests for spiking layers."""

import torch

from spikeflow.layers import SpikingConv2d, SpikingLinear


class TestSpikingLinear:
    def test_forward_shape(self):
        layer = SpikingLinear(128, 256)
        x = torch.randn(2, 10, 128)
        out = layer(x)
        assert out.shape == (2, 10, 256)

    def test_sfa_mode_matches_relu(self):
        layer = SpikingLinear(64, 64, bias=False)
        layer.set_sfa_mode(True)
        x = torch.randn(4, 32, 64)
        out = layer(x)
        linear_out = layer.linear(x)
        assert torch.allclose(out, torch.relu(linear_out))

    def test_has_parameters(self):
        layer = SpikingLinear(128, 256)
        assert layer.linear.weight.shape == (256, 128)
        assert layer.linear.bias.shape == (256,)


class TestSpikingConv2d:
    def test_forward_shape(self):
        layer = SpikingConv2d(3, 64, kernel_size=3, padding=1)
        x = torch.randn(2, 3, 32, 32)
        out = layer(x)
        assert out.shape == (2, 64, 32, 32)
