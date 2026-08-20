# SpikeFlow Documentation

**High-performance Spiking Neural Network library with Transformer-level API and efficiency.**

## Installation

```bash
pip install spikeflow
```

Only dependency: `torch>=2.1.0`

## Quick Start

### Build a Spiking Model

```python
from spikeflow.models import SpikingResNet18
from spikeflow.training import SFATrainer

model = SpikingResNet18(num_classes=10)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
trainer = SFATrainer(model, optimizer)

# Train at transformer speed (T=1, standard backprop)
trainer.train_sfa(train_loader)

# Infer with spike dynamics (T=4, energy-efficient)
trainer.enable_spike_mode(timesteps=4)
results = trainer.evaluate(test_loader)
```

### Train on CIFAR-10

```python
from spikeflow.pipelines.cifar10 import train_cifar10
from spikeflow.models import SpikingResNet18

model = SpikingResNet18(num_classes=10)
results = train_cifar10(model, epochs=100)
print(f"Best accuracy: {results['best_accuracy']:.4f}")
```

### Profile Energy Consumption

```python
from spikeflow.energy import EnergyProfiler

profiler = EnergyProfiler(hardware="gpu_a100")
profile = profiler.profile_snn(model, input_shape=(1, 3, 224, 224))
print(profiler.report(profile))
```

## Neurons

| Module | Description |
|--------|-------------|
| `LIFNode` | Leaky Integrate-and-Fire neuron |
| `IFNode` | Integrate-and-Fire neuron (no leak) |
| `ParametricLIFNode` | LIF with learnable time constant per channel |
| `AdaptiveLIFNode` | LIF with spike-frequency adaptation |
| `IzhikevichNode` | Izhikevich neuron (RS/IB/CH/LTS presets) |

### LIF Dynamics

```
v(t+1) = decay * v(t) + x(t)
spike = (v >= threshold)
if spike: v = v_reset
```

Where `decay = exp(-dt / tau)`.

### SFA Mode

All neurons support SFA (Spike Firing Approximation) mode where they behave as ReLU, enabling T=1 training with standard backprop:

```python
model.set_sfa_mode(True)   # ReLU behavior (training)
model.set_sfa_mode(False)  # Spike dynamics (inference)
```

## Layers

| Module | Description |
|--------|-------------|
| `SpikingLinear` | Drop-in replacement for `nn.Linear` |
| `SpikingConv2d` | Drop-in replacement for `nn.Conv2d` |

Architecture: `Conv/Linear -> LIF Neuron`

## Attention

| Module | Description |
|--------|-------------|
| `SpikingSelfAttention` | XNOR / Rate / Hybrid spike attention |
| `SpikingFFN` | Feed-forward network with spikes |

### XNOR Attention (addition-only, no multiplications)

```python
from spikeflow.attention import SpikingSelfAttention

# Binary XNOR attention
attn = SpikingSelfAttention(d_model=768, n_heads=12, attention_mode="xnor")
```

Three modes:
- `xnor` — Binary XNOR, fastest, lowest energy
- `rate` — Rate-coded, better accuracy
- `hybrid` — XNOR + rate correction, best balance

## Models

| Model | Params | ImageNet | CIFAR-10 |
|-------|--------|----------|----------|
| `SpikingViTTiny` | 5.7M | ~75% | ~91% |
| `SpikingViTSmall` | 22M | ~80% | ~93% |
| `SpikingViTBase` | 87M | ~83% | ~95% |
| `SpikingResNet18` | 11M | ~70% | ~77% (50ep) |
| `SpikingResNet34` | 21M | ~74% | ~95% |
| `SpikingResNet50` | 25M | ~76% | ~96% |
| `SpikingConvNeXtTiny` | 29M | ~82% | ~94% |
| `SpikingConvNeXtSmall` | 50M | ~84% | ~95% |

## Training

### SFATrainer

Core training engine implementing Spike Firing Approximation:

```python
from spikeflow.training import SFATrainer

trainer = SFATrainer(model, optimizer, scheduler=scheduler)
trainer.train_sfa(train_loader)           # T=1, fast training
trainer.enable_spike_mode(timesteps=4)    # Switch to inference
results = trainer.evaluate(test_loader)   # T=4 evaluation
```

### SpikeAdamW

AdamW optimizer with support for binary weight regularization.

### BinaryConnect

Trains with binary (1-bit) weights for 32x compression.

## Quantization

| Module | Description |
|--------|-------------|
| `BinaryWeightQuantizer` | 1-bit weights, 32x compression |
| `SpikeQuantizer` | INT8/INT4 quantization |

## Energy Profiling

Multi-hardware energy profiling for SNN vs ANN comparison:

```python
from spikeflow.energy import EnergyProfiler

profiler = EnergyProfiler(hardware="gpu_a100")
profile = profiler.profile_snn(model, input_shape=(1, 3, 224, 224))
print(profiler.report(profile))
```

Supported hardware: A100, V100, Loihi, ARM Cortex-M4.

## Export

Export to NIR (Neuromorphic Intermediate Representation) for deployment on neuromorphic hardware:

```python
from spikeflow.export import NIRExporter

exporter = NIRExporter()
exporter.export(model, input_shape=(1, 3, 224, 224), path="model.nir")
```

## Pipelines

| Module | Description |
|--------|-------------|
| `train_cifar10` | Complete CIFAR-10 training pipeline |
| `train_imagenet` | Complete ImageNet training pipeline |
| `ann_to_snn` | Convert pretrained ANN to SNN |
| `SpikingDistillation` | Knowledge distillation from ANN teacher |

## How SFA Training Works

Standard SNN training uses Backpropagation Through Time (BPTT):
- Forward pass: compute T timesteps -> T x compute
- Backward pass: propagate through T timesteps -> T x memory

**SFA eliminates this overhead:**

```
Training:  T=1, neurons behave as ReLU  -> same speed as transformer
Inference: T=D, real spike dynamics      -> energy-efficient deployment
```

This works because ReLU is the continuous approximation of spike rate coding.

## License

MIT
