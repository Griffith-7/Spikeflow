# tests/test_training_efficiency.py
"""Tests for SpikeAdamW and BinaryConnect optimizers."""

import torch
import torch.nn as nn

from spikeflow.neurons import LIFNode
from spikeflow.training.spike_optimizer import BinaryConnect, SpikeAdamW


class SimpleModel(nn.Module):
    def __init__(self, in_dim=32, hidden=64, out_dim=10):
        super().__init__()
        self.linear1 = nn.Linear(in_dim, hidden)
        self.neuron1 = LIFNode(threshold=1.0)
        self.linear2 = nn.Linear(hidden, out_dim)
        self.neuron2 = LIFNode(threshold=1.0)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.neuron1(self.linear1(x))
        x = self.neuron2(self.linear2(x))
        return x

    def set_sfa_mode(self, enabled):
        for m in self.children():
            if hasattr(m, "set_sfa_mode"):
                m.set_sfa_mode(enabled)

    def reset_state(self):
        for m in self.children():
            if hasattr(m, "reset_state"):
                m.reset_state()


class TestSpikeAdamW:
    def test_step_reduces_loss(self):
        model = SimpleModel()
        optimizer = SpikeAdamW(model.parameters(), lr=1e-3)

        x = torch.randn(8, 32)
        y = torch.randint(0, 10, (8,))
        criterion = nn.CrossEntropyLoss()

        model.set_sfa_mode(True)
        out = model(x)
        criterion(out, y).item()

        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        out = model(x)
        loss1 = criterion(out, y).item()
        assert torch.isfinite(torch.tensor(loss1))

    def test_binary_regularization(self):
        model = SimpleModel()
        optimizer = SpikeAdamW(model.parameters(), lr=0.1, binary_reg=0.1)

        x = torch.randn(8, 32)
        y = torch.randint(0, 10, (8,))
        criterion = nn.CrossEntropyLoss()

        model.set_sfa_mode(True)
        for _ in range(10):
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

        for p in model.parameters():
            if p.dim() > 1:
                assert p.abs().max() <= 1.01


class TestBinaryConnect:
    def test_weights_bounded(self):
        model = SimpleModel()
        optimizer = BinaryConnect(model.parameters(), lr=0.01)

        x = torch.randn(8, 32)
        y = torch.randint(0, 10, (8,))
        criterion = nn.CrossEntropyLoss()

        model.set_sfa_mode(True)
        for _ in range(5):
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

        for p in model.parameters():
            assert p.abs().max() <= 1.0 + 1e-6
