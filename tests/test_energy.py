# tests/test_energy.py
"""Tests for energy profiler and export tools."""

from spikeflow.energy.profiler import EnergyProfiler
from spikeflow.export.nir import NIRExporter
from spikeflow.models.resnet import SpikingResNet18


class TestEnergyProfiler:
    def test_profile_snn(self):
        model = SpikingResNet18(num_classes=10)
        profiler = EnergyProfiler(hardware="gpu_a100")
        profile = profiler.profile_snn(model, input_shape=(1, 3, 32, 32), timesteps=4)
        assert profile.total_params > 0
        assert profile.total_synops > 0
        assert 0 <= profile.spike_rate <= 1

    def test_report(self):
        model = SpikingResNet18(num_classes=10)
        profiler = EnergyProfiler(hardware="gpu_a100")
        profile = profiler.profile_snn(model, input_shape=(1, 3, 32, 32), timesteps=4)
        report = profiler.report(profile)
        assert "Parameters" in report
        assert "SynOps" in report
        assert "Energy" in report

    def test_compare(self):
        model = SpikingResNet18(num_classes=10)
        profiler = EnergyProfiler(hardware="gpu_a100")
        profile = profiler.profile_snn(model, input_shape=(1, 3, 32, 32), timesteps=4)
        ann_macs = profiler._estimate_macs(model, (1, 3, 32, 32))
        comparison = profiler.compare(profile, ann_macs)
        assert "snn_energy_mj" in comparison
        assert "ann_energy_mj" in comparison
        assert comparison["energy_saving_pct"] > 0  # SNN should be more efficient

    def test_hardware_variants(self):
        model = SpikingResNet18(num_classes=10)
        for hw in ["gpu_a100", "gpu_v100", "loihi_2", "edge_arm"]:
            profiler = EnergyProfiler(hardware=hw)
            profile = profiler.profile_snn(model, input_shape=(1, 3, 32, 32), timesteps=4)
            assert profile.energy_synops_mj > 0


class TestNIRExporter:
    def test_export(self):
        model = SpikingResNet18(num_classes=10)
        exporter = NIRExporter()
        nir_graph = exporter.export(model, input_shape=(1, 3, 32, 32))
        assert len(nir_graph["nodes"]) > 0
        assert len(nir_graph["edges"]) > 0
        assert nir_graph["format"] == "spikeflow-graph"

    def test_summary(self):
        model = SpikingResNet18(num_classes=10)
        exporter = NIRExporter()
        nir_graph = exporter.export(model, input_shape=(1, 3, 32, 32))
        summary = exporter.summary(nir_graph)
        assert "NIR Graph Summary" in summary
        assert "Layers" in summary

    def test_save_load(self, tmp_path):
        model = SpikingResNet18(num_classes=10)
        exporter = NIRExporter()
        nir_graph = exporter.export(model, input_shape=(1, 3, 32, 32))
        path = str(tmp_path / "model.nir")
        exporter.save(nir_graph, path)
        loaded = exporter.load(path)
        assert loaded["format"] == "spikeflow-graph"
        assert len(loaded["nodes"]) == len(nir_graph["nodes"])
