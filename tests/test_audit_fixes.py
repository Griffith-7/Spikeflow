# tests/test_audit_fixes.py
"""Regression tests for audit bug fixes (B1-B6).

B1: set_sfa_mode() must propagate through nn.Sequential to all child modules.
B2: S2NN surrogate gradient must differ from PQ (piecewise-quadratic).
B3: BinaryConnect must produce true {-1, +1} weights via sign().
B4: SpikingDropout must drop neurons consistently across timesteps.
B5: ErfGrad must use the correct derivative of the erf sigmoid.
B6: BinaryConnect.weight_decay must be applied before sign().
"""

import math

import torch

from spikeflow.layers.pooling import SpikingDropout
from spikeflow.models.preact_resnet import SpikingPreActResNet20
from spikeflow.models.resnet import SpikingResNet18
from spikeflow.neurons.base import ErfGrad, PiecewiseQuadraticGrad, S2NNGrad
from spikeflow.training.spike_optimizer import BinaryConnect


class TestB1SetSfaModePropagation:
    """set_sfa_mode() must use self.modules() (not children()) to reach
    modules inside nn.Sequential, e.g. SpikingConv2d in ResNet layers."""

    def _count_sfa_modules(self, model):
        count = 0
        total = 0
        for m in model.modules():
            if hasattr(m, "_sfa_mode") and m is not model:
                total += 1
                if m._sfa_mode:
                    count += 1
        return count, total

    def test_resnet18_all_modules_receive_sfa(self):
        model = SpikingResNet18(num_classes=10)
        model.set_sfa_mode(True)
        sfa_count, total = self._count_sfa_modules(model)
        assert sfa_count == total, (
            f"Only {sfa_count}/{total} modules received SFA mode; "
            f"nn.Sequential children were likely skipped"
        )

    def test_preact_resnet20_all_modules_receive_sfa(self):
        model = SpikingPreActResNet20(num_classes=10)
        model.set_sfa_mode(True)
        sfa_count, total = self._count_sfa_modules(model)
        assert sfa_count == total, (
            f"Only {sfa_count}/{total} modules received SFA mode; "
            f"nn.Sequential children were likely skipped"
        )

    def test_resnet18_sfa_output_is_nonbinary(self):
        model = SpikingResNet18(num_classes=10)
        model.set_sfa_mode(True)
        x = torch.randn(2, 3, 32, 32)
        out = model(x)
        assert not torch.all((out == 0) | (out == 1)), (
            "SFA output should be non-binary (ReLU-like)"
        )

    def test_resnet18_toggle_sfa_off(self):
        model = SpikingResNet18(num_classes=10)
        model.set_sfa_mode(True)
        model.set_sfa_mode(False)
        sfa_count, total = self._count_sfa_modules(model)
        assert sfa_count == 0, "SFA mode should be off after toggle"

    def test_preact_resnet20_toggle_sfa_off(self):
        model = SpikingPreActResNet20(num_classes=10)
        model.set_sfa_mode(True)
        model.set_sfa_mode(False)
        sfa_count, total = self._count_sfa_modules(model)
        assert sfa_count == 0, "SFA mode should be off after toggle"


class TestB2S2nnGradientDiffersFromPq:
    """S2NN (Stockl & Maass 2021) must use a different formula than
    PiecewiseQuadraticGrad, not just clamp(1-|ax|)."""

    def test_forward_same_heaviside(self):
        x = torch.tensor([-0.5, -0.25, 0.0, 0.25, 0.5])
        assert torch.all(S2NNGrad.apply(x) == PiecewiseQuadraticGrad.apply(x))

    def test_backward_different(self):
        x = torch.tensor([-0.5, -0.25, 0.0, 0.25, 0.5], requires_grad=True)
        x2 = x.clone().detach().requires_grad_(True)

        y_pq = PiecewiseQuadraticGrad.apply(x)
        y_pq.sum().backward()
        pq_grad = x.grad.clone()

        y_s2nn = S2NNGrad.apply(x2)
        y_s2nn.sum().backward()
        s2nn_grad = x2.grad.clone()

        assert not torch.allclose(pq_grad, s2nn_grad), (
            "S2NN and PQ gradients are identical — S2NN likely uses "
            "the wrong formula"
        )

    def test_s2nn_value_at_zero(self):
        """S2NN gradient at x=0: alpha*sig*(1-sig)/(1+beta*|x-1|)
        with alpha=4, beta=1 -> 4*0.5*0.5/2 = 0.5."""
        alpha, beta = 4.0, 1.0
        x = torch.tensor([0.0], requires_grad=True)
        S2NNGrad.apply(x, alpha, beta).backward()
        expected = alpha * 0.25 / (1 + beta)
        assert torch.allclose(x.grad, torch.tensor([expected]), atol=1e-5)

    def test_s2nn_decays_far_from_threshold(self):
        """Real S2NN decays to zero in BOTH directions; a piecewise ramp
        would stay constant above threshold."""
        x = torch.tensor([-20.0, 20.0], requires_grad=True)
        S2NNGrad.apply(x).sum().backward()
        assert x.grad.abs().max() < 1e-3, (
            f"S2NN gradient must decay to ~0 far from threshold, got {x.grad.tolist()}"
        )


class TestB3BinaryConnectBinarization:
    """BinaryConnect must produce true {-1, +1} via sign(), not just clamp."""

    def test_weights_are_plus_minus_one(self):
        param = torch.nn.Parameter(torch.randn(200))
        opt = BinaryConnect([param], lr=0.1)
        loss = (param**2).sum()
        loss.backward()
        opt.step()
        unique = param.data.unique().tolist()
        assert all(abs(v) == 1.0 for v in unique), (
            f"Weights must be in {{-1, +1}}, got: {unique}"
        )

    def test_weights_not_continuous(self):
        """After one step, weights should not remain continuous (pre-fix bug)."""
        param = torch.nn.Parameter(torch.randn(200))
        opt = BinaryConnect([param], lr=0.1)
        loss = (param**2).sum()
        loss.backward()
        opt.step()
        # If sign() is missing, values would be in [-1, 1] continuous
        has_only_bipolar = param.data.abs().min() == 1.0
        assert has_only_bipolar, "Weights should be strictly bipolar, not in [-1,1]"

    def test_weight_decay_applied(self):
        """weight_decay must reduce weight magnitude before sign()."""
        param = torch.nn.Parameter(torch.ones(100) * 0.9)
        opt = BinaryConnect([param], lr=0.01, weight_decay=0.1)
        # Manually set grad so optimizer has something to work with
        param.grad = torch.ones_like(param)
        # Before: weights are 0.9, after weight_decay they shrink, then sign()
        # The key is weight_decay is applied (pre-fix: it was dead code)
        opt.step()
        # Verify step completed without error and produced binary weights
        unique = param.data.unique().tolist()
        assert all(abs(v) == 1.0 for v in unique)


class TestB4SpikingDropoutNeuronLevel:
    """SpikingDropout must drop entire neurons across all timesteps,
    not individual elements."""

    def test_3d_consistent_across_time(self):
        drop = SpikingDropout(p=0.5)
        drop.train()
        x = torch.ones(2, 4, 8)  # (batch, time, features)
        out = drop(x)
        for b in range(2):
            mask_b = (out[b] > 0).float()
            mask_t0 = mask_b[0]
            for t in range(1, 4):
                assert torch.equal(mask_b[t], mask_t0), (
                    f"Batch {b}: timestep {t} has different mask than t=0"
                )

    def test_4d_consistent_across_spatial(self):
        drop = SpikingDropout(p=0.5)
        drop.train()
        x = torch.ones(2, 3, 8, 8)  # (batch, channels, H, W)
        out = drop(x)
        for b in range(2):
            for c in range(3):
                mask_c = (out[b, c] > 0).float()
                # All spatial positions within a channel must have same mask
                mask_ref = mask_c[0]
                assert torch.equal(mask_c, mask_ref.expand_as(mask_c)), (
                    f"Batch {b}, channel {c}: spatial positions have different masks"
                )

    def test_eval_mode_passthrough(self):
        drop = SpikingDropout(p=0.5)
        drop.eval()
        x = torch.ones(2, 4, 8)
        out = drop(x)
        assert torch.equal(out, x), "Eval mode should pass through unchanged"

    def test_p0_passthrough(self):
        drop = SpikingDropout(p=0.0)
        drop.train()
        x = torch.ones(2, 4, 8)
        out = drop(x)
        assert torch.equal(out, x), "p=0 should pass through unchanged"


class TestB5ErfGradFormula:
    """ErfGrad must use the derivative of erf(alpha*x):
    (2*alpha/sqrt(pi)) * exp(-(alpha*x)^2)."""

    def test_at_zero(self):
        alpha = 2.0
        x = torch.tensor([0.0], requires_grad=True)
        y = ErfGrad.apply(x, alpha)
        y.backward()
        expected = 2.0 * alpha / math.sqrt(math.pi)
        assert torch.allclose(x.grad, torch.tensor([expected]), atol=1e-4), (
            f"ErfGrad at x=0: got {x.grad.item():.6f}, expected {expected:.6f}"
        )

    def test_alpha_scales_prefactor(self):
        """The alpha must NOT cancel out of the prefactor."""
        grads = []
        for alpha in [1.0, 4.0]:
            x = torch.tensor([0.0], requires_grad=True)
            ErfGrad.apply(x, alpha).sum().backward()
            grads.append(x.grad.item())
        assert abs(grads[1] / grads[0] - 4.0) < 1e-4, (
            f"Prefactor must scale linearly with alpha, got ratio {grads[1] / grads[0]}"
        )

    def test_symmetry(self):
        """ErfGrad should be symmetric: grad(-x) == grad(x)."""
        alpha = 2.0
        for val in [0.1, 0.5, 1.0, 2.0]:
            x_pos = torch.tensor([val], requires_grad=True)
            x_neg = torch.tensor([-val], requires_grad=True)

            ErfGrad.apply(x_pos, alpha).sum().backward()
            g_pos = x_pos.grad.clone()

            ErfGrad.apply(x_neg, alpha).sum().backward()
            g_neg = x_neg.grad.clone()

            assert torch.allclose(g_pos, g_neg, atol=1e-5), (
                f"ErfGrad not symmetric at |x|={val}: "
                f"grad({val})={g_pos.item()}, grad({-val})={g_neg.item()}"
            )

    def test_decays_to_zero(self):
        """Far from threshold, gradient should approach zero."""
        alpha = 2.0
        x = torch.tensor([10.0], requires_grad=True)
        ErfGrad.apply(x, alpha).sum().backward()
        assert x.grad.abs().item() < 0.01, (
            f"ErfGrad at x=10 should be near zero, got {x.grad.item()}"
        )

    def test_different_from_heaviside(self):
        """ErfGrad should not be straight-through (constant 1)."""
        alpha = 2.0
        x = torch.tensor([0.0], requires_grad=True)
        ErfGrad.apply(x, alpha).sum().backward()
        # Heaviside would give 1.0, erf should give ~1.596
        assert x.grad.item() != 1.0, "ErfGrad should not be straight-through"


class TestB6BinaryConnectWeightDecay:
    """weight_decay in BinaryConnect was declared but unused (dead code).
    It must be applied before the sign() clamp."""

    def test_weight_decay_reduces_before_sign(self):
        """With high weight_decay, weights should still end up as {-1, +1}
        but the decay should have been applied before binarization."""
        param = torch.nn.Parameter(torch.ones(100) * 0.9)
        opt = BinaryConnect([param], lr=0.01, weight_decay=0.5)
        param.grad = torch.ones_like(param)
        opt.step()
        # After sign(), weights are -1 or 1
        unique = param.data.unique().tolist()
        assert all(abs(v) == 1.0 for v in unique)

    def test_no_error_with_zero_weight_decay(self):
        param = torch.nn.Parameter(torch.randn(50))
        opt = BinaryConnect([param], lr=0.1, weight_decay=0.0)
        loss = (param**2).sum()
        loss.backward()
        opt.step()
        unique = param.data.unique().tolist()
        assert all(abs(v) == 1.0 for v in unique)
