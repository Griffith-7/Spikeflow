"""Base neuron model with surrogate gradient support."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class Heaviside(torch.autograd.Function):
    """Heaviside step function with straight-through estimator."""

    @staticmethod
    def forward(ctx, x):
        return (x > 0).to(x.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class SigmoidGrad(torch.autograd.Function):
    """Heaviside with sigmoid surrogate gradient."""

    @staticmethod
    def forward(ctx, x, alpha=5.0):
        ctx.save_for_backward(x)
        ctx.alpha = alpha
        return (x > 0).to(x.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        alpha = ctx.alpha
        grad_input = grad_output * alpha / (1 + torch.cosh(alpha * x)).pow(2)
        return grad_input, None


class ATanGrad(torch.autograd.Function):
    """Heaviside with arctangent surrogate gradient."""

    @staticmethod
    def forward(ctx, x, alpha=2.0):
        ctx.save_for_backward(x)
        ctx.alpha = alpha
        return (x > 0).to(x.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        alpha = ctx.alpha
        grad_input = grad_output * alpha / (1 + (alpha * x).pow(2)) / 2
        return grad_input, None


class PiecewiseQuadraticGrad(torch.autograd.Function):
    """Heaviside with piecewise quadratic surrogate gradient (most popular in SNNs)."""

    @staticmethod
    def forward(ctx, x, alpha=1.0):
        ctx.save_for_backward(x)
        ctx.alpha = alpha
        return (x > 0).to(x.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        alpha = ctx.alpha
        ax = alpha * x
        grad_input = torch.where(
            ax.abs() < 1,
            grad_output * (1 - ax.abs()),
            torch.zeros_like(grad_input := grad_output),
        )
        return grad_input, None


class PiecewiseExpGrad(torch.autograd.Function):
    """Heaviside with piecewise exponential surrogate gradient."""

    @staticmethod
    def forward(ctx, x, alpha=1.0):
        ctx.save_for_backward(x)
        ctx.alpha = alpha
        return (x > 0).to(x.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        alpha = ctx.alpha
        grad_input = grad_output * torch.exp(-alpha * x.abs())
        return grad_input, None


class ErfGrad(torch.autograd.Function):
    """Heaviside with error function surrogate gradient.

    Surrogate function: erf(alpha * x), whose derivative is the
    Gaussian (2*alpha/sqrt(pi)) * exp(-(alpha*x)^2).
    """

    @staticmethod
    def forward(ctx, x, alpha=2.0):
        ctx.save_for_backward(x)
        ctx.alpha = alpha
        return (x > 0).to(x.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        alpha = ctx.alpha
        grad_input = (
            grad_output * (2.0 * alpha / math.sqrt(math.pi)) * torch.exp(-((alpha * x) ** 2))
        )
        return grad_input, None


class SuperSpikeGrad(torch.autograd.Function):
    """SuperSpike surrogate (Neftci et al., 2019): β-sigmoid steepness."""

    @staticmethod
    def forward(ctx, x, alpha=5.0, beta=1.0):
        ctx.save_for_backward(x)
        ctx.alpha = alpha
        ctx.beta = beta
        return (x > 0).to(x.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        alpha, beta = ctx.alpha, ctx.beta
        sigmoid = torch.sigmoid(beta * x)
        grad_input = grad_output * alpha * sigmoid * (1 - sigmoid)
        return grad_input, None


class SoftSignGrad(torch.autograd.Function):
    """Heaviside with softsign surrogate gradient."""

    @staticmethod
    def forward(ctx, x, alpha=1.0):
        ctx.save_for_backward(x)
        ctx.alpha = alpha
        return (x > 0).to(x.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        alpha = ctx.alpha
        grad_input = grad_output / (1 + alpha * x.abs()).pow(2)
        return grad_input, None


class LeakyKReLUGrad(torch.autograd.Function):
    """Leaky KReLU surrogate gradient (non-zero below threshold)."""

    @staticmethod
    def forward(ctx, x, alpha=0.1, k=5.0):
        ctx.save_for_backward(x)
        ctx.alpha = alpha
        ctx.k = k
        return (x > 0).to(x.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        alpha, k = ctx.alpha, ctx.k
        grad_input = torch.where(
            x > 0,
            grad_output * k,
            grad_output * alpha,
        )
        return grad_input, None


class S2NNGrad(torch.autograd.Function):
    """S2NN surrogate gradient (Stöckl & Maass, 2021).

    Surrogate function: sigmoid(alpha * x) / (1 + beta * |x - 1|), so the
    surrogate gradient is

        alpha * sigmoid(alpha*x) * (1 - sigmoid(alpha*x)) / (1 + beta*|x-1|)

    It peaks near threshold and decays to zero far from it in BOTH directions
    (unlike a piecewise ramp, which stays constant above threshold).
    """

    @staticmethod
    def forward(ctx, x, alpha=4.0, beta=1.0):
        ctx.save_for_backward(x)
        ctx.alpha = alpha
        ctx.beta = beta
        return (x > 0).to(x.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        alpha, beta = ctx.alpha, ctx.beta
        sig = torch.sigmoid(alpha * x)
        decay = 1.0 / (1.0 + beta * (x - 1.0).abs())
        grad_input = grad_output * alpha * sig * (1.0 - sig) * decay
        return grad_input, None, None


class Threshold(nn.Module):
    """Firing threshold with configurable surrogate gradient."""

    SURROGATES = {
        "heaviside": Heaviside,
        "sigmoid": SigmoidGrad,
        "atan": ATanGrad,
        "pq": PiecewiseQuadraticGrad,
        "pe": PiecewiseExpGrad,
        "erf": ErfGrad,
        "superspike": SuperSpikeGrad,
        "softsign": SoftSignGrad,
        "leaky_krelu": LeakyKReLUGrad,
        "s2nn": S2NNGrad,
    }

    def __init__(self, threshold: float = 1.0, surrogate: str = "sigmoid"):
        super().__init__()
        self.threshold = threshold
        self.surrogate_fn = self.SURROGATES[surrogate]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.surrogate_fn.apply(x - self.threshold)

    def extra_repr(self) -> str:
        return f"threshold={self.threshold}, surrogate={self.surrogate_fn.__name__}"


class BaseNeuron(nn.Module, ABC):
    """Abstract base class for all spiking neurons.

    Every neuron maintains a membrane potential `v` and fires spikes.
    The key design: forward() is called once per timestep inside a loop
    managed by `TenableModule`, keeping the API clean.

    For SFA training (Spike Firing Approximation):
    - During training with T=1, the neuron outputs activations directly
    - During inference, it outputs binary spikes over T timesteps
    """

    def __init__(
        self,
        threshold: float = 1.0,
        surrogate: str = "sigmoid",
        tau: float = 2.0,
        dt: float = 1.0,
        v_reset: float = 0.0,
    ):
        super().__init__()
        self.threshold_module = Threshold(threshold, surrogate)
        self.tau = tau
        self.dt = dt
        self.v_reset = v_reset
        self.v: torch.Tensor | None = None
        self._sfa_mode = False
        self._readout = False

    def reset_state(self):
        """Reset membrane potential to initial state."""
        self.v = None

    @abstractmethod
    def charge(self, x: torch.Tensor) -> torch.Tensor:
        """Update membrane potential based on input."""
        ...

    def fire(self, x: torch.Tensor) -> torch.Tensor:
        """Apply threshold to generate spikes."""
        if self._sfa_mode:
            return x
        return self.threshold_module(x)

    def neuronal_reset(self, spike: torch.Tensor):
        """Reset membrane potential after firing.

        Hard reset: membrane is driven to ``v_reset`` where a spike was
        emitted (identical to the previous reset-to-zero behavior when
        ``v_reset=0``, which is the default).
        """
        if self.v is not None and not self._sfa_mode:
            spike = spike.detach()
            self.v = self.v * (1.0 - spike) + self.v_reset * spike

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.v is None:
            self.v = torch.zeros_like(x)

        self.v = self.charge(x)
        if self._readout:
            return self.v
        spike = self.fire(self.v)
        self.neuronal_reset(spike)
        return spike

    def set_sfa_mode(self, enabled: bool):
        """Enable/disable Spike Firing Approximation mode.

        In SFA mode: no threshold, no reset, no membrane potential.
        The neuron behaves like a standard ReLU, allowing training at T=1
        with standard backprop. At inference, spike trains are generated.
        """
        self._sfa_mode = enabled
        if enabled:
            self.v = None

    def set_readout(self, enabled: bool):
        """Enable readout mode: return membrane potential instead of binary spikes."""
        self._readout = enabled

    def extra_repr(self) -> str:
        return (
            f"tau={self.tau}, dt={self.dt}, "
            f"v_reset={self.v_reset}, sfa={self._sfa_mode}, "
            f"readout={self._readout}"
        )
