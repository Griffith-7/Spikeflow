"""
SpikeFlow: High-performance Spiking Neural Network library.

Core Features:
    - Transformer-like API (SpikingViT, SpikingResNet, SpikingConvNeXt)
    - SFA Training (T=1 train, T=D infer = same speed as transformers)
    - Binary Weight Quantization (1B params = 125MB)
    - XNOR Attention (addition-only, no multiplications)
    - Energy Profiling and NIR Export for neuromorphic deployment
"""

__version__ = "0.1.0"

from spikeflow import (
    attention,
    datasets,
    encoders,
    energy,
    export,
    layers,
    models,
    neurons,
    pipelines,
    quantization,
    training,
    visualization,
)

__all__ = [
    "neurons", "layers", "encoders", "datasets", "visualization",
    "attention", "training", "quantization", "models", "energy",
    "pipelines", "export",
]
