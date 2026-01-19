from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Food-101 is natural images; ImageNet normalization is a solid default.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class Food101DataConfig:
    data_root: str = "data/raw"
    batch_size: int = 64
    num_workers: int = 0
    download: bool = False
    image_size: int = 224


def make_food101_loaders(cfg: Food101DataConfig) -> tuple[DataLoader, DataLoader]:
    train_tf = transforms.Compose(
        [
            transforms.RandomResizedCrop(cfg.image_size, scale=(0.6, 1.0), ratio=(3 / 4, 4 / 3)),
            transforms.RandomHorizontalFlip(),
            transforms.RandAugment(num_ops=2, magnitude=9),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            transforms.RandomErasing(p=0.25, scale=(0.02, 0.2), ratio=(0.3, 3.3), value="random"),
        ]
    )

    test_tf = transforms.Compose(
        [
            transforms.Resize(int(cfg.image_size * 256 / 224)),
            transforms.CenterCrop(cfg.image_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )

    train_ds = datasets.Food101(root=cfg.data_root, split="train", download=cfg.download, transform=train_tf)
    test_ds = datasets.Food101(root=cfg.data_root, split="test", download=cfg.download, transform=test_tf)

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
        batch_size=max(128, cfg.batch_size),
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, test_loader

