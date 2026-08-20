# Contributing to SpikeFlow

Thanks for your interest in contributing! Here's how to get started.

## Setup

```bash
git clone https://github.com/Griffith-7/Spikeflow.git
cd Spikeflow
pip install -e ".[dev]"
```

## Code Quality

All code must pass before merge:

```bash
ruff check spikeflow/     # Linting
ruff format spikeflow/    # Formatting
mypy spikeflow/           # Type checking
pytest tests/ -v          # Tests
```

## Rules (Ponytail Mode)

1. Write only what the task needs
2. No over-engineering
3. Type hints on all public APIs
4. Tests prove behavior, not implementation
5. Shortest diff that solves the problem

## Pull Request Checklist

- [ ] Tests pass (`pytest tests/ -v`)
- [ ] Lint passes (`ruff check`)
- [ ] Types pass (`mypy`)
- [ ] New code has tests
- [ ] Docstrings on public APIs
- [ ] CHANGELOG.md updated

## Project Structure

```
spikeflow/
  neurons/       Spiking neuron models (LIF, IF, LSTM, etc.)
  layers/        Drop-in replacements for nn.Linear, nn.Conv2d, etc.
  attention/     XNOR / Rate / Hybrid spike attention
  models/        Pre-built architectures (ViT, ResNet, ConvNeXt)
  training/      SFA trainer, optimizers, distributed
  encoders/      Input encoding (Poisson, Latency, Population)
  quantization/  Binary and INT8/INT4 weight quantization
  energy/        Hardware energy profiling
  export/        NIR format for neuromorphic deployment
  pipelines/     End-to-end training pipelines
  datasets/      Neuromorphic dataset loaders
tests/           Unit tests
```

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
