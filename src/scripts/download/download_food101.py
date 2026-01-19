import argparse
from pathlib import Path

from torchvision import datasets


def main() -> None:
    p = argparse.ArgumentParser(description="Download Food-101 into data/raw/")
    p.add_argument("--out", type=str, default="data/raw", help="Output directory (dataset root)")
    args = p.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # This will create out_dir/food-101/ (plus a downloaded archive during fetch).
    datasets.Food101(root=str(out_dir), split="train", download=True)
    datasets.Food101(root=str(out_dir), split="test", download=True)

    print(f"Downloaded Food-101 to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
