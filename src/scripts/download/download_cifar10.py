import argparse
from pathlib import Path

from torchvision import datasets


def main() -> None:
    p = argparse.ArgumentParser(description="Download CIFAR-10 into data/raw/")
    p.add_argument("--out", type=str, default="data/raw", help="Output directory (dataset root)")
    args = p.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # This will create out_dir/cifar-10-batches-py/ (plus a downloaded archive during fetch).
    datasets.CIFAR10(root=str(out_dir), train=True, download=True)
    datasets.CIFAR10(root=str(out_dir), train=False, download=True)

    print(f"Downloaded CIFAR-10 to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
