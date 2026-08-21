# SpikeFlow

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-red.svg)](https://pytorch.org)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/Griffith-7/Spikeflow/pulls)
[![Tests](https://img.shields.io/badge/tests-45%20passing-brightgreen.svg)]()

**Train SNNs at transformer speed. Deploy with binary spike efficiency.**

> **Status:** Alpha. API is stable. Benchmarks pending — accuracy and speed numbers below are targets based on architecture design, not measured results.

```
pip install spikeflow
```

---

## What makes SpikeFlow different

| Capability | SpikingJelly | snnTorch | Norse | **SpikeFlow** |
|---|---|---|---|---|
| Train at transformer speed (T=1) | ❌ | ❌ | ❌ | ✅ **SFA** |
| Binary weights (1-bit) | ❌ | ❌ | ❌ | ✅ **BinaryConnect** |
| Sign/XNOR-style binary attention | ❌ | ❌ | ❌ | ✅ **simulated popcount** |
| Single dependency (`torch>=2.1.0`) | ❌ torch+cupy+triton | ❌ torch+tonic | ❌ torch+pytest | ✅ **torch + numpy** |
| Surrogate gradients | 10+ | 3 | 4 | **10** (all unique) |
| Exact gradients (IFT + saltation) | ❌ | ❌ | ❌ | ✅ **spikeflow.exact** |
| Neuron models | 15+ | 9 | 6 | **7** (LIF, IF, PLIF, ALIF, Izhikevich, LSTM, RNN) |
| Layer types | 20+ | 5 | 10+ | **10** (Conv1d/2d/3d, ConvTranspose, Linear, Pool, Dropout, Voting) |
| CIFAR-10 (ResNet18) | 95.6% (BPTT T=4) | 57% | — | *target: ~90%* |
| ImageNet (ResNet18) | 69-70% | — | — | *target: ~69%* |
| Neuromorphic datasets | 13 built-in | 2 | — | **5** (DVS, N-MNIST, SHD, SSC) |
| ANN-to-SNN conversion | recipe-based | basic | — | **recipe + BN fold + threshold calib** |
| Mixed precision (FP16/BF16) | ✅ | ❌ | ❌ | ✅ |
| Distributed training | ✅ | ❌ | ❌ | ✅ (FSDP/DDP) |
| NIR export | ❌ | ✅ | ❌ | ✅ |
| Energy profiling | ❌ | ❌ | ❌ | ✅ (A100/V100/Loihi/ARM) |

## Quick Start

```python
import torch
from spikeflow.models import SpikingResNet18
from spikeflow.training import SFATrainer

model = SpikingResNet18(num_classes=10)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
trainer = SFATrainer(model, optimizer, device="cuda")

# Train at transformer speed (T=1, standard backprop)
trainer.train_sfa(train_loader)

# Infer with spike dynamics (T=D, energy-efficient)
trainer.enable_spike_mode(timesteps=4)
results = trainer.evaluate(test_loader)
```

## How SFA Works

```
Standard SNN training:  T=4 loops -> 4x slower than transformers
SpikeFlow SFA:          T=1 train -> same speed as transformers
                        T=D infer -> energy-efficient deployment
```

```
Training (T=1):   x -> [Linear -> ReLU] -> loss     <- standard backprop
Inference (T=D):  x -> [Linear -> LIF] -> spike train <- binary, energy-efficient
                  ^ same weights, different neuron mode
```

## Architecture

```
spikeflow/
  neurons/       LIF, IF, ParametricLIF, AdaptiveLIF, Izhikevich, LSTM, RNN
  layers/        Conv1d/2d/3d, ConvTranspose, Linear, MaxPool, AvgPool, Dropout
  attention/     XNOR / Rate / Hybrid self-attention, SpikingFFN
  models/        SpikingViT, SpikingResNet, SpikingConvNeXt, PreActResNet
  training/      SFA trainer, EMA, mixed precision, distributed (FSDP/DDP)
  encoders/      Poisson, Latency (TTFS), Population coding
  quantization/  Binary (1-bit), INT8, INT4
  energy/        Multi-hardware energy profiler
  export/        NIR format for neuromorphic deployment
  exact/         Exact IFT gradients + saltation matrices (from Exact-SNN)
  pipelines/     CIFAR-10, ImageNet, ANN-to-SNN, knowledge distillation
  datasets/      DVS128, N-MNIST, CIFAR10-DVS, SHD, SSC (via tonic)
```

## Neurons

| Module | Description | Dynamics |
|--------|-------------|----------|
| `LIFNode` | Leaky Integrate-and-Fire | v' = decay*v + x |
| `IFNode` | Integrate-and-Fire (no leak) | v' = v + x |
| `ParametricLIFNode` | LIF with learnable tau per channel | learnable decay |
| `AdaptiveLIFNode` | Spike-frequency adaptation | threshold increases after spikes |
| `IzhikevichNode` | Izhikevich (RS/IB/CH/LTS) | v' = 0.04v^2 + 5v + 140 - u + I |
| `SpikingLSTM` | LSTM with spiking membrane | gates + LIF readout |

## Surrogate Gradients

All 10 surrogates are mathematically unique:

| Name | Formula | Best for |
|------|---------|----------|
| `sigmoid` | alpha / (1 + cosh(alpha*x))^2 | general (default) |
| `pq` | piecewise linear: 1 - |alpha*x| | most popular in SNN literature |
| `erf` | (2*alpha/sqrt(pi)) * exp(-(alpha*x)^2) | smooth gradients |
| `superspike` | beta * sigmoid(beta*x) * (1 - sigmoid(beta*x)) | Neftci et al. 2019 |
| `s2nn` | sigmoid-derivative scaled by 1/(1+β\|x−1\|) (Stockl & Maass 2021) | decays both sides of threshold |
| `atan` | alpha / (1 + (alpha*x)^2) / 2 | fast approximation |
| `pe` | exp(-alpha * |x|) | exponential decay |
| `softsign` | 1 / (1 + alpha*|x|)^2 | bounded gradient |
| `leaky_krelu` | k above, alpha below threshold | non-zero below threshold |
| `heaviside` | straight-through (constant 1) | simple baseline |

## Exact Gradients (spikeflow.exact)

Mathematically exact gradients using Implicit Function Theorem and saltation matrices — no surrogate approximation. Forked from [Exact-SNN](https://github.com/Griffith-7/Exact-snn).

```python
from spikeflow.exact import TTFSNet

net = TTFSNet([784, 128, 10])
loss, grads, t_out = net.loss_and_grads(t_in, y)
```

## Encoders

```python
from spikeflow.encoders import PoissonEncoder, LatencyEncoder, PopulationEncoder

# Rate coding: P(spike) = input intensity
encoder = PoissonEncoder(timesteps=4)
spikes = encoder(x)  # (T, batch, C, H, W)

# Latency coding (TTFS): stronger inputs fire earlier
encoder = LatencyEncoder(timesteps=4)
spikes = encoder(x)

# Population coding: Gaussian-tuned neurons
encoder = PopulationEncoder(n_neurons=4, timesteps=4)
spikes = encoder(x)  # (T, batch, C*N)
```

## Models

| Model | Params | Status |
|-------|--------|--------|
| `SpikingResNet18` | 11M | implemented, needs benchmarking |
| `SpikingPreActResNet20` | 0.27M | implemented, needs benchmarking |
| `SpikingViTTiny` | 5.7M | implemented, needs benchmarking |
| `SpikingViTSmall` | 22M | implemented, needs benchmarking |
| `SpikingViTBase` | 87M | implemented, needs benchmarking |
| `SpikingConvNeXtTiny` | 29M | implemented, needs benchmarking |

## Energy Profiling

```python
from spikeflow.energy import EnergyProfiler

profiler = EnergyProfiler(hardware="gpu_a100")
profile = profiler.profile_snn(model, input_shape=(1, 3, 224, 224))
print(profiler.report(profile))
```

Supports: A100, V100, Loihi, ARM Cortex-M4.

## NIR Export

```python
from spikeflow.export import NIRExporter

exporter = NIRExporter()
graph = exporter.export(model, input_shape=(1, 3, 224, 224))
exporter.save(graph, "model.json")
```

Exports a NIR-*inspired* JSON graph (sequential topology). This is not yet
the standards-compliant NIR format — for Loihi/Lava deployment, convert via
the official `nir` package.

## Mixed Precision + EMA

```python
from spikeflow.training import SFATrainer

trainer = SFATrainer(
    model, optimizer,
    use_mixed_precision=True,  # FP16 on CUDA
    ema_decay=0.999,           # exponential moving average
)
```

## Distributed Training

```python
from spikeflow.training.distributed import setup_distributed, wrap_ddp

setup_distributed(rank=0, world_size=2)
model = wrap_ddp(model, rank=0)
```

## ANN-to-SNN Conversion

```python
from spikeflow.pipelines.ann2snn_recipe import convert_ann_to_snn, ConversionRecipe

recipe = ConversionRecipe(timesteps=32, fold_bn=True, calibrate_batches=100)
snn = convert_ann_to_snn(pretrained_ann, recipe, data_loader=calib_loader)
```

## Install

```bash
pip install spikeflow
# Or from source:
git clone https://github.com/Griffith-7/Spikeflow.git
cd Spikeflow && pip install -e ".[dev]"
```

Only dependencies: `torch>=2.1.0`, `numpy`

## How It Works

1. **SFA Training** — Neurons behave as ReLU during training (T=1), standard backprop. Same speed as transformers.
2. **Spike Inference** — Neurons use real LIF dynamics at inference (T=D timesteps). Binary spikes replace multiplications.
3. **Binary Quantization** — BinaryConnect optimizer clamps weights to {-1, +1} after each step. Sign-mode attention simulates XNOR-popcount similarity.
4. **Exact Gradients** — Optional IFT + saltation matrix gradients for TTFS networks (spikeflow.exact).

## Citation

```bibtex
@software{spikeflow2026,
  title = {SpikeFlow: High-performance Spiking Neural Network Library},
  year = {2026},
  url = {https://github.com/Griffith-7/Spikeflow},
  license = {MIT}
}
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

MIT — see [LICENSE](LICENSE).
