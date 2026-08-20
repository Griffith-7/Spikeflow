"""Spiking convolution layers (1D, 2D, 3D, Transpose)."""

from __future__ import annotations

import torch.nn as nn
from torch import Tensor

from spikeflow.neurons.lif import LIFNode


class SpikingConv1d(nn.Module):
    """Spiking 1D convolution: Conv1d -> LIF."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int],
        stride: int | tuple[int] = 1,
        padding: int | tuple[int] = 0,
        dilation: int | tuple[int] = 1,
        groups: int = 1,
        bias: bool = True,
        threshold: float = 1.0,
        surrogate: str = "sigmoid",
        tau: float = 2.0,
        readout: bool = False,
    ):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, dilation=dilation,
            groups=groups, bias=bias,
        )
        self.neuron = LIFNode(threshold=threshold, surrogate=surrogate, tau=tau)
        if readout:
            self.neuron.set_readout(True)

    def forward(self, x: Tensor) -> Tensor:
        return self.neuron(self.conv(x))

    def reset_state(self):
        self.neuron.reset_state()

    def set_sfa_mode(self, enabled: bool):
        self.neuron.set_sfa_mode(enabled)

    def set_readout(self, enabled: bool):
        self.neuron.set_readout(enabled)


class SpikingConv2d(nn.Module):
    """Drop-in replacement for nn.Conv2d that outputs spikes.

    Architecture: Conv2d -> LIF Neuron
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        stride: int | tuple[int, int] = 1,
        padding: int | tuple[int, int] = 0,
        dilation: int | tuple[int, int] = 1,
        groups: int = 1,
        bias: bool = True,
        threshold: float = 1.0,
        surrogate: str = "sigmoid",
        tau: float = 2.0,
        readout: bool = False,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, dilation=dilation,
            groups=groups, bias=bias,
        )
        self.neuron = LIFNode(
            threshold=threshold,
            surrogate=surrogate,
            tau=tau,
        )
        if readout:
            self.neuron.set_readout(True)

    def forward(self, x: Tensor) -> Tensor:
        return self.neuron(self.conv(x))

    def reset_state(self):
        self.neuron.reset_state()

    def set_sfa_mode(self, enabled: bool):
        self.neuron.set_sfa_mode(enabled)

    def set_readout(self, enabled: bool):
        self.neuron.set_readout(enabled)

    def extra_repr(self) -> str:
        return f"{self.conv}, sfa={self.neuron._sfa_mode}"


class SpikingConv3d(nn.Module):
    """Spiking 3D convolution: Conv3d -> LIF."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int, int],
        stride: int | tuple[int, int, int] = 1,
        padding: int | tuple[int, int, int] = 0,
        dilation: int | tuple[int, int, int] = 1,
        groups: int = 1,
        bias: bool = True,
        threshold: float = 1.0,
        surrogate: str = "sigmoid",
        tau: float = 2.0,
        readout: bool = False,
    ):
        super().__init__()
        self.conv = nn.Conv3d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, dilation=dilation,
            groups=groups, bias=bias,
        )
        self.neuron = LIFNode(threshold=threshold, surrogate=surrogate, tau=tau)
        if readout:
            self.neuron.set_readout(True)

    def forward(self, x: Tensor) -> Tensor:
        return self.neuron(self.conv(x))

    def reset_state(self):
        self.neuron.reset_state()

    def set_sfa_mode(self, enabled: bool):
        self.neuron.set_sfa_mode(enabled)

    def set_readout(self, enabled: bool):
        self.neuron.set_readout(enabled)

    def extra_repr(self) -> str:
        return f"{self.conv}, sfa={self.neuron._sfa_mode}"


class SpikingConvTranspose2d(nn.Module):
    """Spiking transposed 2D convolution: ConvTranspose2d -> LIF."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        stride: int | tuple[int, int] = 1,
        padding: int | tuple[int, int] = 0,
        output_padding: int | tuple[int, int] = 0,
        groups: int = 1,
        bias: bool = True,
        dilation: int | tuple[int, int] = 1,
        threshold: float = 1.0,
        surrogate: str = "sigmoid",
        tau: float = 2.0,
        readout: bool = False,
    ):
        super().__init__()
        self.conv = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, output_padding=output_padding,
            groups=groups, bias=bias, dilation=dilation,
        )
        self.neuron = LIFNode(threshold=threshold, surrogate=surrogate, tau=tau)
        if readout:
            self.neuron.set_readout(True)

    def forward(self, x: Tensor) -> Tensor:
        return self.neuron(self.conv(x))

    def reset_state(self):
        self.neuron.reset_state()

    def set_sfa_mode(self, enabled: bool):
        self.neuron.set_sfa_mode(enabled)

    def set_readout(self, enabled: bool):
        self.neuron.set_readout(enabled)

    def extra_repr(self) -> str:
        return f"{self.conv}, sfa={self.neuron._sfa_mode}"
