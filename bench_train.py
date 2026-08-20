"""Real training + benchmark on CIFAR-10. ponytail: one script, all results."""
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from spikeflow.models.resnet import SpikingResNet18
from spikeflow.training.sfa import SFATrainer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def make_loaders(data_dir="./data", batch_size=128, workers=0):
    import torchvision
    import torchvision.transforms as T

    train_tf = T.Compose([
        T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(),
        T.ToTensor(), T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    test_tf = T.Compose([
        T.ToTensor(), T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    train_set = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=False, transform=train_tf)
    test_set = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=False, transform=test_tf)
    return (
        DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True),
        DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True),
    )


@torch.no_grad()
def eval_spike(model, loader, device, timesteps=4):
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        for m in model.modules():
            if hasattr(m, "reset_state"):
                m.reset_state()
        acc = None
        for _ in range(timesteps):
            out = model(x)
            out = torch.nan_to_num(out)
            acc = out if acc is None else acc + out
        correct += acc.argmax(1).eq(y).sum().item()
        total += y.size(0)
    return correct / total


def train_and_benchmark(epochs=20, batch_size=128, lr=3e-3):
    print(f"\n{'='*65}")
    print(f" SpikeFlow CIFAR-10 Benchmark — {DEVICE.upper()}")
    print(f"{'='*65}")

    train_loader, test_loader = make_loaders(batch_size=batch_size)
    model = SpikingResNet18(num_classes=10).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f" Model: SpikingResNet18 | Params: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    trainer = SFATrainer(model, optimizer, scheduler=scheduler, device=DEVICE, grad_clip=1.0, use_mixed_precision=(DEVICE == "cuda"))

    # Warmup batch for timing
    print(f"\n Training {epochs} epochs (SFA mode, T=1)...")
    print(f" {'Epoch':>5} {'Time':>6} {'Loss':>7} {'Train':>6} {'Test':>6} {'Best':>6} {'img/s':>8}")
    print(f" {'-'*5} {'-'*6} {'-'*7} {'-'*6} {'-'*6} {'-'*6} {'-'*8}")

    best = 0.0
    total_train_time = 0.0
    total_train_images = 0

    for epoch in range(epochs):
        t0 = time.time()
        trainer.enable_sfa_mode()
        model.train()
        ep_loss = ep_correct = ep_total = 0

        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            ep_loss += loss.item() * x.size(0)
            ep_correct += out.argmax(1).eq(y).sum().item()
            ep_total += x.size(0)

        scheduler.step()
        train_time = time.time() - t0
        total_train_time += train_time
        total_train_images += ep_total

        # Evaluate spike mode
        trainer.enable_spike_mode(timesteps=4)
        test_acc = eval_spike(model, test_loader, DEVICE, timesteps=4)
        best = max(best, test_acc)
        imgs_per_sec = ep_total / train_time

        print(
            f" {epoch+1:5d} {train_time:5.1f}s "
            f"{ep_loss/ep_total:7.4f} {ep_correct/ep_total:6.4f} "
            f"{test_acc:6.4f} {best:6.4f} {imgs_per_sec:8.0f}"
        )

    # === Benchmark: SFA vs Spike inference speed ===
    print(f"\n{'='*65}")
    print(" Inference Speed Benchmark")
    print(f"{'='*65}")

    dummy = torch.randn(32, 3, 32, 32, device=DEVICE)
    model.eval()

    # SFA inference (T=1)
    trainer.enable_sfa_mode()
    torch.cuda.synchronize() if DEVICE == "cuda" else None
    t0 = time.time()
    for _ in range(50):
        model(dummy)
    torch.cuda.synchronize() if DEVICE == "cuda" else None
    sfa_ms = (time.time() - t0) / 50 * 1000

    # Spike inference (T=4)
    trainer.enable_spike_mode(timesteps=4)
    for m in model.modules():
        if hasattr(m, "reset_state"):
            m.reset_state()
    torch.cuda.synchronize() if DEVICE == "cuda" else None
    t0 = time.time()
    for _ in range(50):
        for m in model.modules():
            if hasattr(m, "reset_state"):
                m.reset_state()
        acc = None
        for _ in range(4):
            out = model(dummy)
            acc = out if acc is None else acc + out
    torch.cuda.synchronize() if DEVICE == "cuda" else None
    spike4_ms = (time.time() - t0) / 50 * 1000

    # Spike inference (T=8)
    trainer.enable_spike_mode(timesteps=8)
    torch.cuda.synchronize() if DEVICE == "cuda" else None
    t0 = time.time()
    for _ in range(50):
        for m in model.modules():
            if hasattr(m, "reset_state"):
                m.reset_state()
        acc = None
        for _ in range(8):
            out = model(dummy)
            acc = out if acc is None else acc + out
    torch.cuda.synchronize() if DEVICE == "cuda" else None
    spike8_ms = (time.time() - t0) / 50 * 1000

    mem_mb = torch.cuda.max_memory_allocated() / 1024**2 if DEVICE == "cuda" else 0

    print(f" SFA (T=1) train+infer:  {sfa_ms:7.2f} ms/batch")
    print(f" Spike (T=4) infer:      {spike4_ms:7.2f} ms/batch  ({spike4_ms/sfa_ms:.2f}x SFA)")
    print(f" Spike (T=8) infer:      {spike8_ms:7.2f} ms/batch  ({spike8_ms/sfa_ms:.2f}x SFA)")

    # === ANN vs SNN comparison ===
    ann_model = nn.Sequential(
        nn.Flatten(), nn.Linear(3*32*32, 256), nn.ReLU(),
        nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 10),
    ).to(DEVICE)

    dummy_flat = dummy.flatten(1)
    torch.cuda.synchronize() if DEVICE == "cuda" else None
    t0 = time.time()
    for _ in range(50):
        ann_model(dummy_flat)
    torch.cuda.synchronize() if DEVICE == "cuda" else None
    ann_ms = (time.time() - t0) / 50 * 1000

    print(f"\n{'='*65}")
    print(" Summary")
    print(f"{'='*65}")
    print(" Model:           SpikingResNet18")
    print(f" Parameters:      {n_params:,} ({n_params*4/1024**2:.1f} MB FP32, {n_params/8/1024**2:.1f} MB binary)")
    print(f" Best Accuracy:   {best:.4f} (CIFAR-10, {epochs} epochs)")
    print(f" Training Speed:  {total_train_images/total_train_time:.0f} img/s (SFA, T=1)")
    print(f" Peak GPU Mem:    {mem_mb:.0f} MB")
    print(f" Inference SFA:   {sfa_ms:.1f} ms/batch")
    print(f" Inference T=4:   {spike4_ms:.1f} ms/batch")
    print(f" ANN baseline:    {ann_ms:.1f} ms/batch")
    print(" SFA Training = standard backprop (same as transformer training)")
    print(f"{'='*65}\n")

    return {"best_accuracy": best, "params": n_params, "train_speed_img_s": total_train_images/total_train_time}


if __name__ == "__main__":
    train_and_benchmark(epochs=50)
