"""SpikeFlow Pipelines — complete training pipelines for standard benchmarks."""

from spikeflow.pipelines.cifar10 import train_cifar10
from spikeflow.pipelines.convert import ann_to_snn
from spikeflow.pipelines.imagenet import train_imagenet

__all__ = ["train_cifar10", "train_imagenet", "ann_to_snn"]
