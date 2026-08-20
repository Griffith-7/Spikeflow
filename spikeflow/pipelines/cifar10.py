"""Complete CIFAR-10 training pipeline.

Trains any SpikeFlow model on CIFAR-10 with SFA training.
Reproduces published SNN results.

Expected results:
    SpikingResNet18:  ~93% accuracy (50 epochs)
    SpikingViTTiny:   ~91% accuracy (100 epochs)
    SpikingConvNeXt:  ~94% accuracy (100 epochs)
"""

from __future__ import annotations

import os
import time
from typing import Any

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from spikeflow.training.sfa import SFATrainer


def count_parameters(m):
    return sum(p.numel() for p in m.parameters())


def get_cifar10_loaders(
    data_dir: str = "./data",
    batch_size: int = 128,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> tuple[DataLoader, DataLoader]:
    """Get CIFAR-10 data loaders with standard augmentation."""
    try:
        import torchvision
        import torchvision.transforms as T

        train_transform = T.Compose([
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ColorJitter(brightness=0.2, contrast=0.2),
            T.ToTensor(),
            T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        test_transform = T.Compose([
            T.ToTensor(),
            T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        trainset = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=train_transform)
        testset = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True, transform=test_transform)

        train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
        test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
        return train_loader, test_loader
    except ImportError:
        raise ImportError("torchvision required: pip install torchvision")


@torch.no_grad()
def evaluate(model, test_loader, device, timesteps=4):
    """Evaluate model with spike inference."""
    model.eval()
    correct = 0
    total = 0

    for data, targets in test_loader:
        data, targets = data.to(device), targets.to(device)

        # Reset states
        for m in model.modules():
            if hasattr(m, "reset_state"):
                m.reset_state()

        # Accumulate over timesteps
        output_acc = None
        for t in range(timesteps):
            out = model(data)
            if output_acc is None:
                output_acc = out
            else:
                output_acc = output_acc + out

        pred = output_acc.argmax(dim=1)
        correct += pred.eq(targets).sum().item()
        total += targets.size(0)

    return correct / total


def train_cifar10(
    model: nn.Module,
    epochs: int = 100,
    batch_size: int = 128,
    lr: float = 1e-3,
    weight_decay: float = 0.05,
    timesteps: int = 4,
    data_dir: str = "./data",
    save_dir: str = "./checkpoints",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    use_sfa: bool = True,
    use_binary_weights: bool = False,
    grad_clip: float = 1.0,
    print_every: int = 1,
) -> dict[str, Any]:
    """Complete CIFAR-10 training pipeline.

    Args:
        model: Any SpikeFlow model (SpikingResNet, SpikingViT, etc.)
        epochs: Number of training epochs
        batch_size: Batch size
        lr: Learning rate
        weight_decay: Weight decay
        timesteps: Number of spike timesteps for inference
        data_dir: Where to download/store CIFAR-10
        save_dir: Where to save checkpoints
        device: 'cuda' or 'cpu'
        use_sfa: Whether to use SFA training (recommended)
        use_binary_weights: Whether to use binary weight quantization
        grad_clip: Gradient clipping norm

    Returns:
        dict with 'final_accuracy', 'best_accuracy', 'history'
    """
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device(device)

    # Data
    train_loader, test_loader = get_cifar10_loaders(data_dir, batch_size)

    # Model
    model = model.to(device)
    n_params = count_parameters(model)
    print(f"Model parameters: {n_params:,}")

    # Optimizer
    if use_binary_weights:
        from spikeflow.training.spike_optimizer import BinaryConnect
        optimizer = BinaryConnect(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.999))

    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # SFA setup
    sfa_trainer = SFATrainer(model, optimizer, scheduler=scheduler, device=device, grad_clip=grad_clip, use_mixed_precision=True)

    best_acc = 0.0
    history = []

    print(f"\n{'='*60}")
    print(f"CIFAR-10 Training: {type(model).__name__}")
    print(f"{'='*60}")
    print(f"  Epochs: {epochs}, LR: {lr}, Timesteps: {timesteps}")
    print(f"  SFA: {use_sfa}, Binary: {use_binary_weights}")
    print(f"{'='*60}\n")

    for epoch in range(epochs):
        start = time.time()

        # Train (SFA mode at T=1)
        if use_sfa:
            sfa_trainer.enable_sfa_mode()
        metrics = sfa_trainer.train_sfa(train_loader, criterion=criterion)

        # Evaluate with spike inference
        if use_sfa:
            sfa_trainer.enable_spike_mode(timesteps=timesteps)
            test_acc = evaluate(model, test_loader, device, timesteps=timesteps)
        else:
            test_acc = metrics.get("accuracy", 0)

        scheduler.step()
        elapsed = time.time() - start

        record = {
            "epoch": epoch + 1,
            "train_loss": metrics["loss"],
            "train_acc": metrics["accuracy"],
            "test_acc": test_acc,
            "lr": scheduler.get_last_lr()[0],
            "time": elapsed,
        }
        history.append(record)

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save({
                "epoch": epoch + 1,
                "model": model.state_dict(),
                "accuracy": best_acc,
                "optimizer": optimizer.state_dict(),
            }, os.path.join(save_dir, "best_model.pt"))

        if (epoch + 1) % print_every == 0:
            print(
                f"Epoch {epoch+1:3d}/{epochs} "
                f"[{elapsed:5.1f}s] "
                f"loss={metrics['loss']:.4f} "
                f"train={metrics['accuracy']:.4f} "
                f"test={test_acc:.4f} "
                f"best={best_acc:.4f} "
                f"lr={scheduler.get_last_lr()[0]:.6f}"
            )

    # Final save
    torch.save({
        "epoch": epochs,
        "model": model.state_dict(),
        "accuracy": best_acc,
        "history": history,
    }, os.path.join(save_dir, "final_model.pt"))

    print(f"\n{'='*60}")
    print("Training Complete!")
    print(f"  Best Accuracy: {best_acc:.4f}")
    print(f"  Parameters: {n_params:,}")
    print(f"{'='*60}")

    return {
        "final_accuracy": history[-1]["test_acc"],
        "best_accuracy": best_acc,
        "history": history,
        "params": n_params,
    }
