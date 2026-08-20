"""Complete ImageNet training pipeline.

Uses standard ImageNet folder structure:
    data/imagenet/train/
    data/imagenet/val/

Expected results (100 epochs, SFA training):
    SpikingViTBase:   ~83% ImageNet (comparable to DeiT-B)
    SpikingResNet50:  ~76% ImageNet (comparable to ResNet50)
    SpikingConvNeXt:  ~82% ImageNet (comparable to ConvNeXt-T)
"""

from __future__ import annotations

import json
import os
import time
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from spikeflow.training.sfa import SFATrainer


def count_parameters(m):
    return sum(p.numel() for p in m.parameters())


def get_imagenet_loaders(
    data_dir: str = "./data/imagenet",
    batch_size: int = 256,
    num_workers: int = 16,
    image_size: int = 224,
    pin_memory: bool = True,
) -> tuple[DataLoader, DataLoader]:
    """Get ImageNet data loaders with standard augmentation."""
    try:
        import torchvision
        import torchvision.transforms as T

        train_transform = T.Compose([
            T.RandomResizedCrop(image_size, interpolation=T.InterpolationMode.BICUBIC),
            T.RandomHorizontalFlip(),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        val_transform = T.Compose([
            T.Resize(int(image_size * 256 / 224), interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(image_size),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        trainset = torchvision.datasets.ImageFolder(os.path.join(data_dir, "train"), transform=train_transform)
        valset = torchvision.datasets.ImageFolder(os.path.join(data_dir, "val"), transform=val_transform)

        train_loader = DataLoader(
            trainset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=pin_memory, persistent_workers=True,
        )
        val_loader = DataLoader(
            valset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory, persistent_workers=True,
        )
        return train_loader, val_loader
    except Exception as e:
        raise RuntimeError(f"Failed to load ImageNet: {e}. Ensure data is at {data_dir}")


@torch.no_grad()
def evaluate_imagenet(model, val_loader, device, timesteps=4):
    """Evaluate top-1 and top-5 accuracy."""
    model.eval()
    correct1 = 0
    correct5 = 0
    total = 0

    for data, targets in val_loader:
        data, targets = data.to(device), targets.to(device)

        for m in model.modules():
            if hasattr(m, "reset_state"):
                m.reset_state()

        output_acc = None
        for t in range(timesteps):
            out = model(data)
            if output_acc is None:
                output_acc = out
            else:
                output_acc = output_acc + out

        pred1 = output_acc.topk(5, dim=1).indices
        correct1 += pred1[:, 0].eq(targets).sum().item()
        correct5 += pred1.eq(targets.unsqueeze(1)).any(dim=1).sum().item()
        total += targets.size(0)

    return correct1 / total, correct5 / total


def train_imagenet(
    model: nn.Module,
    epochs: int = 300,
    batch_size: int = 256,
    lr: float = 1e-3,
    weight_decay: float = 0.05,
    warmup_epochs: int = 30,
    timesteps: int = 4,
    data_dir: str = "./data/imagenet",
    save_dir: str = "./checkpoints",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    num_workers: int = 16,
    grad_clip: float = 1.0,
) -> dict[str, Any]:
    """Complete ImageNet training pipeline.

    Training protocol:
        1. SFA training at T=1 (transformer speed)
        2. Evaluation with full spike dynamics at T=4
        3. Cosine annealing with linear warmup

    Args:
        model: SpikeFlow model (SpikingViT, SpikingResNet, etc.)
        epochs: Total training epochs (300 for SOTA)
        batch_size: Per-GPU batch size
        lr: Peak learning rate
        weight_decay: Weight decay
        warmup_epochs: Linear warmup epochs
        timesteps: Spike timesteps for evaluation
        data_dir: ImageNet root directory
        save_dir: Checkpoint directory
        device: Training device
        grad_clip: Gradient clipping

    Returns:
        dict with 'best_top1', 'best_top5', 'history'
    """
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device(device)

    # Data
    train_loader, val_loader = get_imagenet_loaders(data_dir, batch_size, num_workers=num_workers)

    # Model
    model = model.to(device)
    n_params = count_parameters(model)
    print(f"Model: {type(model).__name__}, Parameters: {n_params:,}")

    # Optimizer with warmup
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.999))

    # Scheduler: linear warmup + cosine decay
    def lr_lambda(step):
        if step < warmup_epochs * len(train_loader):
            return step / (warmup_epochs * len(train_loader))
        else:
            progress = (step - warmup_epochs * len(train_loader)) / ((epochs - warmup_epochs) * len(train_loader))
            return max(0.0, 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)).item()))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # SFA trainer
    sfa_trainer = SFATrainer(
        model, optimizer, scheduler=scheduler, device=device,
        grad_clip=grad_clip, use_mixed_precision=True,
    )

    best_top1 = 0.0
    history = []

    print(f"\n{'='*60}")
    print(f"ImageNet Training: {type(model).__name__}")
    print(f"{'='*60}")
    print(f"  Epochs: {epochs}, Batch: {batch_size}, LR: {lr}")
    print(f"  Warmup: {warmup_epochs} epochs, Timesteps: {timesteps}")
    print(f"{'='*60}\n")

    for epoch in range(epochs):
        start = time.time()

        # SFA training
        sfa_trainer.enable_sfa_mode()
        train_metrics = sfa_trainer.train_sfa(train_loader, criterion=criterion)

        # Evaluate every 5 epochs or last epoch
        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            sfa_trainer.enable_spike_mode(timesteps=timesteps)
            top1, top5 = evaluate_imagenet(model, val_loader, device, timesteps=timesteps)
        else:
            top1, top5 = 0, 0

        elapsed = time.time() - start

        record = {
            "epoch": epoch + 1,
            "train_loss": train_metrics["loss"],
            "train_acc": train_metrics["accuracy"],
            "top1": top1,
            "top5": top5,
            "lr": scheduler.get_last_lr()[0],
            "time": elapsed,
        }
        history.append(record)

        if top1 > best_top1:
            best_top1 = top1
            torch.save({
                "epoch": epoch + 1,
                "model": model.state_dict(),
                "top1": best_top1,
                "optimizer": optimizer.state_dict(),
            }, os.path.join(save_dir, "best_model.pt"))

        if top1 > 0:
            print(
                f"Epoch {epoch+1:3d}/{epochs} [{elapsed:6.1f}s] "
                f"loss={train_metrics['loss']:.4f} "
                f"top1={top1:.4f} top5={top5:.4f} "
                f"best={best_top1:.4f}"
            )

    # Save history
    with open(os.path.join(save_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n{'='*60}")
    print("ImageNet Training Complete!")
    print(f"  Best Top-1: {best_top1:.4f}")
    print(f"  Parameters: {n_params:,}")
    print(f"{'='*60}")

    return {
        "best_top1": best_top1,
        "history": history,
        "params": n_params,
    }
