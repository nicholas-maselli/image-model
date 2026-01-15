from __future__ import annotations

from dataclasses import dataclass
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


@dataclass(frozen=True)
class Cifar10DataConfig:
    data_root: str = "data/raw"
    batch_size: int = 128
    num_workers: int = 0
    download: bool = False


def make_cifar10_loaders(cfg: Cifar10DataConfig) -> tuple[DataLoader, DataLoader]:
    train_tf = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )
    test_tf = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )

    train_ds = datasets.CIFAR10(root=cfg.data_root, train=True, download=cfg.download, transform=train_tf)
    test_ds = datasets.CIFAR10(root=cfg.data_root, train=False, download=cfg.download, transform=test_tf)

    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=max(256, cfg.batch_size),
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, test_loader
