# tests/test_bugfixes.py
"""Regression tests for the 2026 audit fixes.

Covers:
    - SFATrainer no longer toggles readout globally (#1)
    - BinaryWeightQuantizer pack/unpack roundtrip + padding (#2)
    - LatencyEncoder non-temporal path (#3)
    - S2NN surrogate decays far from threshold (#4)
    - ErfGrad prefactor scales with alpha (#5)
    - v_reset is honored in neuronal_reset (#6)
    - ParametricLIF tau constrained positive (#14)
    - BinaryConnect skips norm/bias params when given a module (#13)
    - EMA apply_shadow/restore guard
    - SpikeQuantizer consistent forward API
    - CPU-safe mixed precision default (#11)
"""

import math
import warnings

import pytest
import torch
import torch.nn as nn

from spikeflow.encoders import LatencyEncoder
from spikeflow.layers.linear import SpikingLinear
from spikeflow.neurons.base import ErfGrad, S2NNGrad
from spikeflow.neurons.lif import LIFNode, ParametricLIFNode
from spikeflow.quantization.spike_quant import BinaryWeightQuantizer, SpikeQuantizer
from spikeflow.training.sfa import EMA, SFATrainer
from spikeflow.training.spike_optimizer import BinaryConnect


class TestSpikeModeEmitsSpikes:
    """#1: enable_spike_mode must NOT turn every neuron into an analog
    readout — hidden neurons must emit binary spikes and a head configured
    with readout=True must keep its configuration."""

    def _build(self):
        hidden = SpikingLinear(4, 8)
        head = SpikingLinear(8, 3, readout=True)
        model = nn.Sequential(hidden, head)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        trainer = SFATrainer(model, opt, device="cpu", use_mixed_precision=False)
        return model, hidden, head, trainer

    def test_hidden_layer_emits_binary_spikes_after_spike_mode(self):
        model, hidden, _, trainer = self._build()
        trainer.enable_spike_mode(timesteps=4)
        x = torch.rand(2, 4)
        out = hidden(x)
        assert torch.all((out == 0) | (out == 1)), (
            f"Hidden layer must emit binary spikes after enable_spike_mode(), "
            f"got continuous values: {torch.unique(out)[:5].tolist()}..."
        )

    def test_head_readout_config_preserved_by_sfa_toggle(self):
        _, _, head, trainer = self._build()
        trainer.enable_sfa_mode()
        assert head.neuron._readout is True, (
            "enable_sfa_mode() must not clobber the head's readout=True"
        )
        trainer.enable_spike_mode(timesteps=4)
        assert head.neuron._readout is True

    def test_hidden_neurons_not_in_readout_after_spike_mode(self):
        _, hidden, _, trainer = self._build()
        trainer.enable_spike_mode(timesteps=4)
        assert hidden.neuron._readout is False


class TestBinaryWeightQuantizerRoundtrip:
    """#2: pack/unpack must be exact inverses, for any shape."""

    def test_roundtrip_exact_multiple_of_8(self):
        q = BinaryWeightQuantizer()
        w = torch.randn(16, 8)
        restored = q.unpack_binary(q.pack_binary(w), w.shape)
        assert torch.equal(w.sign(), restored)

    def test_roundtrip_non_multiple_of_8(self):
        q = BinaryWeightQuantizer()
        w = torch.randn(3, 3)  # 9 elements -> needs padding
        packed = q.pack_binary(w)
        restored = q.unpack_binary(packed, w.shape)
        assert torch.equal(w.sign(), restored)

    def test_roundtrip_conv_shape(self):
        q = BinaryWeightQuantizer()
        w = torch.randn(64, 32, 3, 3)
        restored = q.unpack_binary(q.pack_binary(w), w.shape)
        assert torch.equal(w.sign(), restored)


class TestLatencyEncoderNonTemporal:
    """#3: temporal_output=False used to scatter into the batch dim."""

    def test_non_temporal_no_crash_and_shape(self):
        enc = LatencyEncoder(timesteps=4, temporal_output=False)
        out = enc(torch.rand(2, 10))  # batch=2 < T-1=3 previously crashed
        assert out.shape == (2, 10)

    def test_temporal_one_spike_per_element(self):
        enc = LatencyEncoder(timesteps=4, temporal_output=True)
        x = torch.rand(3, 5)
        spikes = enc(x)
        assert spikes.shape == (4, 3, 5)
        assert torch.all(spikes.sum(dim=0) == 1), "each element fires exactly once"

    def test_strong_input_fires_earlier(self):
        enc = LatencyEncoder(timesteps=8, temporal_output=False)
        times = enc(torch.tensor([[1.0, 0.0]]))
        assert times[0, 0] == 0.0  # x=1 fires at t=0
        assert times[0, 1] == 7.0  # x=0 fires at t=T-1


class TestSurrogateMath:
    """#4/#5: S2NN decays both directions; erf alpha scales the prefactor."""

    def test_s2nn_decays_to_zero(self):
        x = torch.tensor([-20.0, 20.0], requires_grad=True)
        S2NNGrad.apply(x).sum().backward()
        assert x.grad.abs().max() < 1e-3

    def test_s2nn_not_constant_above_threshold(self):
        x = torch.tensor([0.5, 1.0, 5.0], requires_grad=True)
        S2NNGrad.apply(x).sum().backward()
        grads = x.grad.tolist()
        assert grads[2] < grads[0] * 0.1, "gradient must shrink well above threshold"

    def test_erf_alpha_scales_prefactor(self):
        g = []
        for alpha in [1.0, 4.0]:
            x = torch.tensor([0.0], requires_grad=True)
            ErfGrad.apply(x, alpha).sum().backward()
            g.append(x.grad.item())
        assert abs(g[1] / g[0] - 4.0) < 1e-4

    def test_surrogate_forward_preserves_dtype(self):
        x16 = torch.tensor([-0.5, 0.5], dtype=torch.float16)
        for fn in (ErfGrad, S2NNGrad):
            assert fn.apply(x16).dtype == torch.float16


class TestVResetHonored:
    """#6: v_reset was accepted but never applied."""

    def test_reset_to_v_reset(self):
        node = LIFNode(v_reset=0.5)
        node.v = torch.tensor([1.5])
        node.neuronal_reset(torch.tensor([1.0]))
        assert torch.allclose(node.v, torch.tensor([0.5]))

    def test_default_reset_to_zero_unchanged(self):
        node = LIFNode()  # v_reset=0.0 default
        node.v = torch.tensor([1.5])
        node.neuronal_reset(torch.tensor([1.0]))
        assert torch.allclose(node.v, torch.tensor([0.0]))

    def test_no_spike_no_reset(self):
        node = LIFNode(v_reset=0.5)
        node.v = torch.tensor([1.5])
        node.neuronal_reset(torch.tensor([0.0]))
        assert torch.allclose(node.v, torch.tensor([1.5]))


class TestParametricLIFTauConstraint:
    """#14: negative tau_param must not produce decay > 1."""

    def test_negative_raw_tau_gives_decay_leq_one(self):
        node = ParametricLIFNode(channels=4)
        with torch.no_grad():
            node.tau_param.copy_(torch.tensor([-5.0, -1.0, 0.0, 3.0]))
        assert (node.decay <= 1.0).all()

    def test_init_tau_matches_requested(self):
        node = ParametricLIFNode(channels=4, tau_init=2.0)
        expected_decay = math.exp(-1.0 / 2.0)
        assert torch.allclose(
            node.decay, torch.full((4,), expected_decay), atol=1e-5
        )


class TestBinaryConnectFiltering:
    """#13: passing a module must exclude norm/bias parameters."""

    def test_module_input_skips_norm_and_bias(self):
        model = nn.Sequential(nn.Linear(8, 8), nn.BatchNorm1d(8), nn.Linear(8, 4))
        opt = BinaryConnect(model, lr=0.01)
        binarized_ids = {id(p) for g in opt.param_groups for p in g["params"]}
        all_ids = {id(p) for p in model.parameters()}
        excluded = all_ids - binarized_ids
        bn_weight_id = id(model[1].weight)
        linear_bias_id = id(model[0].bias)
        assert bn_weight_id in excluded
        assert linear_bias_id in excluded
        # The actual weights of Linear layers ARE included.
        assert id(model[0].weight) in binarized_ids
        assert len(binarized_ids) > 0

    def test_all_binarizable_excluded_raises(self):
        model = nn.BatchNorm1d(8)
        with pytest.raises(ValueError):
            BinaryConnect(model)

    def test_raw_list_still_binarizes_everything(self):
        p = torch.nn.Parameter(torch.randn(100))
        opt = BinaryConnect([p], lr=0.1)
        p.grad = torch.ones_like(p)
        opt.step()
        assert all(abs(v) == 1.0 for v in p.data.unique().tolist())


class TestEMAGuard:
    def test_double_apply_shadow_raises(self):
        model = nn.Linear(4, 4)
        ema = EMA(model, decay=0.9)
        ema.apply_shadow()
        with pytest.raises(RuntimeError):
            ema.apply_shadow()

    def test_restore_roundtrip(self):
        model = nn.Linear(4, 4)
        ema = EMA(model, decay=0.5)
        with torch.no_grad():
            for p in model.parameters():
                p.add_(1.0)
        ema.update()
        # Snapshot right before apply_shadow: restore() must return to THIS
        # state (the pre-evaluation weights), not the construction state.
        before_apply = {n: p.clone() for n, p in model.named_parameters()}
        ema.apply_shadow()
        assert all(
            not torch.equal(p, before_apply[n]) for n, p in model.named_parameters()
        ), "apply_shadow should swap in EMA weights"
        ema.restore()
        for n, p in model.named_parameters():
            assert torch.equal(p, before_apply[n])


class TestSpikeQuantizerAPI:
    def test_forward_returns_tensor_in_both_modes(self):
        q = SpikeQuantizer(bits=8)
        w = torch.randn(8, 8)
        q.train()
        out_train = q(w)
        q.eval()
        out_eval = q(w)
        assert isinstance(out_train, torch.Tensor)
        assert isinstance(out_eval, torch.Tensor)
        assert out_train.shape == w.shape

    def test_quantize_returns_int_and_scale(self):
        q = SpikeQuantizer(bits=8)
        w = torch.randn(8, 8)
        wq, scale = q.quantize(w)
        assert wq.dtype == torch.int8
        assert scale > 0
        reconstructed = wq.float() * scale
        assert (reconstructed - w).abs().max() < scale


class TestCPUMixedPrecisionDefault:
    """#11: use_mixed_precision=True on CPU must disable AMP, not break."""

    def test_trainer_cpu_with_amp_flag(self):
        model = nn.Sequential(SpikingLinear(4, 8), SpikingLinear(8, 3, readout=True))
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            trainer = SFATrainer(model, opt, device="cpu")  # default amp=True
        assert any("mixed precision" in str(w.message).lower() for w in caught)
        assert trainer.scaler is None

    def test_cpu_training_step_runs(self):
        model = nn.Sequential(SpikingLinear(4, 8), SpikingLinear(8, 3, readout=True))
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        trainer = SFATrainer(model, opt, device="cpu")  # amp requested on CPU
        x, y = torch.rand(8, 4), torch.randint(0, 3, (8,))
        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(x, y), batch_size=4
        )
        metrics = trainer.train_sfa(loader)
        assert "loss" in metrics
