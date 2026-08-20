# tests/test_neurons.py
"""Tests for spiking neuron models."""

import torch

from spikeflow.neurons import IFNode, LIFNode, ParametricLIFNode
from spikeflow.neurons.adlif import AdaptiveLIFNode
from spikeflow.neurons.izhikevich import IzhikevichNode


class TestLIFNode:
    def test_forward_shape(self):
        node = LIFNode(threshold=1.0)
        x = torch.randn(2, 10)
        out = node(x)
        assert out.shape == (2, 10)

    def test_spike_is_binary(self):
        node = LIFNode(threshold=1.0)
        x = torch.ones(4, 8) * 5.0  # Large input should fire
        out = node(x)
        assert torch.all((out == 0) | (out == 1))

    def test_sfa_mode(self):
        node = LIFNode(threshold=1.0)
        node.set_sfa_mode(True)
        x = torch.randn(2, 10)
        out = node(x)
        # In SFA mode, output should be ReLU(x)
        expected = torch.relu(x)
        assert torch.allclose(out, expected)

    def test_reset_state(self):
        node = LIFNode(threshold=1.0)
        x = torch.randn(2, 10)
        node(x)
        assert node.v is not None
        node.reset_state()
        assert node.v is None

    def test_readout_mode(self):
        node = LIFNode(threshold=1.0)
        node.set_readout(True)
        x = torch.ones(2, 10) * 0.5
        out = node(x)
        assert out.shape == (2, 10)
        # Readout returns continuous membrane potential, not binary
        assert not torch.all((out == 0) | (out == 1))


class TestIFNode:
    def test_forward_shape(self):
        node = IFNode()
        x = torch.randn(2, 10)
        out = node(x)
        assert out.shape == (2, 10)

    def test_no_leak(self):
        """IF neuron should accumulate without decay."""
        node = IFNode(threshold=100.0)  # High threshold to prevent firing
        x = torch.ones(1, 5)
        for _ in range(10):
            node(x)
        # v should be ~10 (no decay)
        assert node.v is not None
        assert torch.allclose(node.v, torch.ones(1, 5) * 10, atol=0.1)


class TestParametricLIF:
    def test_learnable_tau(self):
        node = ParametricLIFNode(channels=16)
        assert node.tau_param.requires_grad
        assert node.tau_param.shape == (16,)

    def test_different_tau_per_channel(self):
        node = ParametricLIFNode(channels=4, tau_init=2.0)
        node.tau_param.data = torch.tensor([1.0, 2.0, 5.0, 10.0])
        decay = node.decay
        # Different taus should produce different decays
        assert decay[0] != decay[1]
        assert decay[0] < decay[2]  # Lower tau = faster decay = smaller decay factor


class TestAdaptiveLIF:
    def test_forward_shape(self):
        node = AdaptiveLIFNode()
        x = torch.randn(2, 10)
        out = node(x)
        assert out.shape == (2, 10)

    def test_adaptation_increases(self):
        node = AdaptiveLIFNode(threshold=1.0, delta_a=0.1)
        x = torch.ones(1, 5) * 5.0
        node(x)
        node(x)
        # Adaptation should have increased after spikes
        assert node.a is not None
        assert node.a.abs().sum() > 0

    def test_reset_clears_adaptation(self):
        node = AdaptiveLIFNode()
        x = torch.ones(1, 5) * 5.0
        node(x)
        node.reset_state()
        assert node.v is None
        assert node.a is None

    def test_sfa_mode(self):
        node = AdaptiveLIFNode()
        node.set_sfa_mode(True)
        x = torch.randn(2, 10)
        out = node(x)
        assert torch.allclose(out, torch.relu(x))


class TestIzhikevich:
    def test_forward_shape(self):
        node = IzhikevichNode(preset="rs")
        x = torch.randn(2, 10)
        out = node(x)
        assert out.shape == (2, 10)

    def test_spike_is_binary(self):
        node = IzhikevichNode(preset="rs")
        x = torch.ones(4, 8) * 20.0
        out = node(x)
        assert torch.all((out == 0) | (out == 1))

    def test_presets(self):
        for preset in IzhikevichNode.PRESETS:
            node = IzhikevichNode(preset=preset)
            x = torch.randn(2, 10)
            out = node(x)
            assert out.shape == (2, 10)

    def test_sfa_mode(self):
        node = IzhikevichNode()
        node.set_sfa_mode(True)
        x = torch.randn(2, 10)
        out = node(x)
        assert torch.allclose(out, torch.relu(x))

    def test_reset_clears_state(self):
        node = IzhikevichNode()
        x = torch.randn(2, 10)
        node(x)
        node.reset_state()
        assert node.v is None
        assert node.u is None
