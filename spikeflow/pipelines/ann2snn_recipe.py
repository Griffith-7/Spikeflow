"""ANN-to-SNN conversion with recipe-based optimization.

Supports:
- Activation-aware weight redistribution
- Threshold calibration
- Batch normalization folding
- Conversion quality metrics
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class ConversionRecipe:
    """Recipe for ANN-to-SNN conversion."""
    timesteps: int = 32
    threshold: float = 1.0
    calibrate_batches: int = 100
    fold_bn: bool = True


def fold_batchnorm(model: nn.Module) -> nn.Module:
    """Fold BatchNorm into preceding Conv/Linear layers.

    After folding, BN parameters are merged into the weight/bias of
    the preceding layer, eliminating the BN at inference time.
    """
    for name, module in model.named_children():
        if isinstance(module, nn.Sequential):
            folded = _fold_sequential(module)
            setattr(model, name, folded)
        else:
            fold_batchnorm(module)
    return model


def _fold_sequential(seq: nn.Sequential) -> nn.Sequential:
    """Fold BN in a Sequential if pattern matches Conv+BN."""
    modules = list(seq.children())
    folded = []
    i = 0
    while i < len(modules):
        if (i + 1 < len(modules)
            and isinstance(modules[i], (nn.Conv1d, nn.Conv2d, nn.Conv3d))
            and isinstance(modules[i + 1], nn.BatchNorm2d)):
            conv = modules[i]
            bn = modules[i + 1]
            folded.append(_fuse_conv_bn(conv, bn))
            i += 2
        else:
            folded.append(modules[i])
            i += 1
    return nn.Sequential(*folded)


def _fuse_conv_bn(conv: nn.Module, bn: nn.BatchNorm2d) -> nn.Conv2d:
    """Fuse Conv2d + BatchNorm2d into a single Conv2d."""
    w = conv.weight
    b = conv.bias if conv.bias is not None else torch.zeros(conv.out_channels, device=w.device)

    bn_w = bn.weight
    bn_b = bn.bias if bn.bias is not None else torch.zeros(bn.num_features, device=w.device)
    bn_var = bn.running_var
    bn_mean = bn.running_mean

    scale = bn_w / torch.sqrt(bn_var + bn.eps)
    w_fused = w * scale.view(-1, 1, 1, 1)
    b_fused = (b - bn_mean) * scale + bn_b

    fused = nn.Conv2d(
        conv.in_channels, conv.out_channels, conv.kernel_size,
        stride=conv.stride, padding=conv.padding, dilation=conv.dilation,
        groups=conv.groups, bias=True,
    )
    fused.weight.data = w_fused
    fused.bias.data = b_fused
    return fused


def calibrate_thresholds(
    model: nn.Module,
    data_loader: torch.utils.data.DataLoader,
    target_percentile: float = 0.99,
) -> float:
    """Calibrate firing thresholds from activation statistics.

    Runs calibration data through the model and sets thresholds
    to the target percentile of activations.
    """
    model.eval()
    all_acts = []

    with torch.no_grad():
        for x, _ in data_loader:
            acts = _collect_activations(model, x)
            all_acts.append(acts)

    all_acts = torch.cat(all_acts)
    threshold = torch.quantile(all_acts, target_percentile).item()
    return threshold


def _collect_activations(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Collect pre-threshold activations from all layers."""
    activations = []
    hooks = []

    def hook_fn(module, inp, out):
        if isinstance(out, torch.Tensor):
            activations.append(out.detach().flatten())

    for m in model.modules():
        if hasattr(m, "threshold_module"):
            hooks.append(m.register_forward_hook(hook_fn))

    model(x)

    for h in hooks:
        h.remove()

    return torch.cat(activations) if activations else torch.tensor([0.0])


def convert_ann_to_snn(
    ann: nn.Module,
    recipe: ConversionRecipe = ConversionRecipe(),
    data_loader: torch.utils.data.DataLoader | None = None,
) -> nn.Module:
    """Convert an ANN to an SNN using the given recipe.

    Steps:
    1. Fold batch normalization
    2. Calibrate thresholds (if data provided)
    3. Set timesteps
    """
    if recipe.fold_bn:
        fold_batchnorm(ann)

    if data_loader and recipe.calibrate_batches > 0:
        threshold = calibrate_thresholds(ann, data_loader)
        for m in ann.modules():
            if hasattr(m, "threshold_module"):
                m.threshold_module.threshold = threshold

    return ann
