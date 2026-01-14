from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def main() -> None:
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2470, 0.2435, 0.2616)),
    ])

    train_ds = datasets.CIFAR10(
        root="data/raw",
        train=True,
        download=False,   # you already downloaded it
        transform=train_tf,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=128,
        shuffle=True,
        num_workers=0,    # Windows-friendly; increase later
        pin_memory=True,
    )

    images, labels = next(iter(train_loader))
    print(f"train size: {len(train_ds)}")
    print(f"classes: {train_ds.classes}")
    print(f"batch images: {images.shape} {images.dtype}")
    print(f"batch labels: {labels.shape} {labels.dtype} (min={labels.min().item()} max={labels.max().item()})")

if __name__ == "__main__":
    main()
