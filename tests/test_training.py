# tests/test_training.py
"""Tests for SFA training."""

import torch
import torch.nn as nn

from spikeflow.neurons import LIFNode
from spikeflow.training.sfa import SFATrainer


class SimpleSNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(32, 64)
        self.neuron1 = LIFNode(threshold=1.0)
        self.linear2 = nn.Linear(64, 10)
        self.neuron2 = LIFNode(threshold=1.0)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.neuron1(self.linear1(x))
        x = self.neuron2(self.linear2(x))
        return x


class TestSFA:
    def test_sfa_enables_relu_like_behavior(self):
        model = SimpleSNN()
        model.eval()
        x = torch.randn(4, 32)

        # Without SFA
        model(x)

        # With SFA
        model.neuron1.set_sfa_mode(True)
        model.neuron2.set_sfa_mode(True)
        out_sfa = model(x)

        # SFA should produce non-binary outputs (like ReLU)
        assert not torch.all((out_sfa == 0) | (out_sfa == 1))

    def test_sfa_training_step(self):
        model = SimpleSNN()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        trainer = SFATrainer(model, optimizer, device="cpu", use_mixed_precision=False)

        # Create simple dataset
        x = torch.randn(16, 32)
        y = torch.randint(0, 10, (16,))
        dataset = torch.utils.data.TensorDataset(x, y)
        loader = torch.utils.data.DataLoader(dataset, batch_size=4)

        metrics = trainer.train_sfa(loader)
        assert "loss" in metrics
        assert "accuracy" in metrics
        assert 0 <= metrics["accuracy"] <= 1
