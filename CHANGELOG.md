# Changelog

All notable changes to SpikeFlow will be documented in this file.

## [0.1.0] — 2026-08-20

### Added
- **Neurons**: LIF, IF, ParametricLIF, AdaptiveLIF, Izhikevich with 10 surrogate gradients
- **Recurrent**: SpikingLSTM, SpikingRNN for temporal sequence tasks
- **Layers**: Conv1d/2d/3d, ConvTranspose2d, Linear, MaxPool, AvgPool, Dropout, VotingLayer
- **Attention**: XNOR, Rate, and Hybrid spike-driven attention
- **Models**: SpikingViT (Tiny/Small/Base), SpikingResNet (18/34/50), SpikingPreActResNet (20/56/110), SpikingConvNeXt (Tiny/Small)
- **Training**: SFA trainer with EMA, mixed precision (FP16/BF16), SpikeAdamW, BinaryConnect
- **Encoders**: Poisson, Latency (TTFS), Population coding
- **Quantization**: Binary (1-bit, 32x compression), INT8, INT4 weight quantization
- **Energy**: Multi-hardware energy profiler (A100/V100/Loihi/ARM)
- **Export**: NIR format export for neuromorphic deployment
- **Pipelines**: CIFAR-10 and ImageNet pipelines, ANN-to-SNN conversion with BN folding and threshold calibration, knowledge distillation
- **Datasets**: DVS128, N-MNIST, CIFAR10-DVS, SHD, SSC (via tonic)
- **Distributed**: DDP + FSDP support with gradient accumulation
- **Step mode**: Single-step (s) and multi-step (m) computation wrapper
- **Visualization**: Spike raster plots, membrane potential traces, firing rate monitoring
- **CI/CD**: GitHub Actions with Python 3.10/3.11/3.12, ruff, mypy, pytest
- **Tests**: 45 unit tests covering all modules
