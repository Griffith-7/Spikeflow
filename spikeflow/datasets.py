"""Neuromorphic dataset loaders for event-based vision and audio benchmarks.

Supports: DVS128 Gesture, N-MNIST, CIFAR10-DVS, SHD, SSC.
Requires optional dependency: tonic.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from torch.utils.data import Dataset


class TonicDataset(Dataset):
    """Generic wrapper around tonic datasets with transform support."""

    def __init__(
        self,
        tonic_cls: str,
        root: str = "./data",
        subdir: str = "",
        train: bool = True,
        transform: Any = None,
        target_transform: Any = None,
        **kwargs,
    ):
        try:
            import tonic
        except ImportError:
            raise ImportError("Install tonic: pip install tonic")
        self.transform = transform
        self.target_transform = target_transform
        cls = getattr(tonic.datasets, tonic_cls)
        init_kwargs = {"root": str(Path(root) / subdir)}
        if hasattr(cls, "train"):
            init_kwargs["train"] = train
        init_kwargs.update(kwargs)
        self.dataset = cls(**init_kwargs)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> tuple:
        events, label = self.dataset[idx]
        if self.transform:
            events = self.transform(events)
        if self.target_transform:
            label = self.target_transform(label)
        return events, label


def DVSGesture(root="./data", train=True, **kw):
    return TonicDataset("DVSGesture", root, subdir="dvs128", train=train, **kw)

def NMNIST(root="./data", train=True, **kw):
    return TonicDataset("NMNIST", root, subdir="nmnist", train=train, **kw)

def CIFAR10DVS(root="./data", train=True, **kw):
    return TonicDataset("CIFAR10DVS", root, subdir="cifar10dvs", train=train, **kw)

def SHD(root="./data", train=True, **kw):
    return TonicDataset("SHD", root, subdir="shd", train=train, **kw)

def SSC(root="./data", train=True, **kw):
    return TonicDataset("SSC", root, subdir="ssc", train=train, **kw)


_DATASET_MAP = {
    "dvs128": DVSGesture, "dvs_gesture": DVSGesture,
    "n-mnist": NMNIST, "nmnist": NMNIST,
    "cifar10-dvs": CIFAR10DVS, "cifar10dvs": CIFAR10DVS,
    "shd": SHD, "ssc": SSC,
}


def get_neuromorphic_dataset(name: str, root: str = "./data", **kwargs) -> Dataset:
    """Factory function to get neuromorphic datasets by name."""
    if name.lower() not in _DATASET_MAP:
        raise ValueError(f"Unknown dataset: {name}. Available: {list(_DATASET_MAP.keys())}")
    return _DATASET_MAP[name.lower()](root=root, **kwargs)
