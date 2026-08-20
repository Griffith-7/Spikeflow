# tests/test_pipelines.py
"""Tests for training pipelines."""

import torch
import torch.nn as nn

from spikeflow.pipelines.convert import SpikingDistillation, ann_to_snn


class TestANNtoSNN:
    def test_conversion(self):
        # Create a simple ANN
        ann = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
        )
        snn = ann_to_snn(ann, timesteps=4)
        # Should still work
        x = torch.randn(2, 32)
        out = snn(x)
        assert out.shape == (2, 10)

    def test_preserves_weights(self):
        ann = nn.Linear(32, 10)
        snn = ann_to_snn(ann, timesteps=1)
        # Weights should be preserved
        assert torch.allclose(ann.weight, snn.weight)


class TestDistillation:
    def test_distillation_loss(self):
        teacher = nn.Linear(32, 10)
        student = nn.Sequential(nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, 10))

        distiller = SpikingDistillation(teacher, student, temperature=4.0)

        x = torch.randn(4, 32)
        with torch.no_grad():
            teacher_out = teacher(x)
        student_out = student(x)

        loss = distiller.distillation_loss(student_out, teacher_out)
        assert loss.item() > 0
