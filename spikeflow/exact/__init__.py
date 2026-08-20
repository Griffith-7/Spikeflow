"""Exact-gradient training for Spiking Neural Networks.

Train SNNs with mathematically exact gradients instead of surrogate gradients.
Solves the non-differentiable spike problem using IFT, saltation matrices,
and escape-noise -- so spike networks train as well as dense networks.

Quick start:
    from spikeflow.exact import TTFSNet
    net = TTFSNet([784, 128, 10])

Fast event-driven engine:
    from spikeflow.exact import EventTTFSNet
    net = EventTTFSNet([784, 128, 10])

Extended architectures:
    from spikeflow.exact.extended import ConvTTFSLayer, SNNConvNet, MultiSpikeNet
"""

__version__ = "1.1.0"

from spikeflow.exact.core import (
    TTFSNetTorch,
    backward_layer_saltation,
    backward_layer_torch,
    backward_multispike_layer,
    backward_multispike_layer_torch,
    device,
    edge_peak_guard,
    forward_layer_torch,
    forward_multispike_layer,
    forward_multispike_layer_torch,
    peak_margin_torch,
)
from spikeflow.exact.event import EventTTFSNet
from spikeflow.exact.extended import kaiming_init, xavier_init
from spikeflow.exact.losses import (
    latency_cross_entropy,
    rate_latency_loss,
    spike_count_cross_entropy,
)
from spikeflow.exact.optim import AdamTorch
from spikeflow.exact.reset import ResetLIF

TTFSNet = TTFSNetTorch

__all__ = [
    "TTFSNet",
    "TTFSNetTorch",
    "EventTTFSNet",
    "forward_layer_torch",
    "backward_layer_torch",
    "backward_layer_saltation",
    "peak_margin_torch",
    "edge_peak_guard",
    "device",
    "ResetLIF",
    "forward_multispike_layer",
    "forward_multispike_layer_torch",
    "backward_multispike_layer",
    "backward_multispike_layer_torch",
    "latency_cross_entropy",
    "spike_count_cross_entropy",
    "rate_latency_loss",
    "AdamTorch",
    "xavier_init",
    "kaiming_init",
]
