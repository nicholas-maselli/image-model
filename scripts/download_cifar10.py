from torchvision import datasets, transforms


def main() -> None:
    transform = transforms.Compose([transforms.ToTensor()])

    train_ds = datasets.CIFAR10(
        root="data/raw",
        train=True,
        download=True,
        transform=transform,
    )
    test_ds = datasets.CIFAR10(
        root="data/raw",
        train=False,
        download=True,
        transform=transform,
    )

    print("Downloaded CIFAR-10")
    print(f"Train size: {len(train_ds)}")
    print(f"Test size:  {len(test_ds)}")
    print(f"Classes:   {train_ds.classes}")

    x, y = train_ds[0]
    print(f"Example: x.shape={tuple(x.shape)} dtype={x.dtype} min={x.min().item():.3f} max={x.max().item():.3f} y={y}")


if __name__ == "__main__":
    main()
