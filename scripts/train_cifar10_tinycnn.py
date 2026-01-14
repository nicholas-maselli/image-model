import argparse
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# If your project isn't packaged yet, run with:
#   Windows PowerShell: $env:PYTHONPATH="src"; uv run python scripts/train_cifar10_tinycnn.py
#   Git Bash:          PYTHONPATH=src uv run python scripts/train_cifar10_tinycnn.py
from imagemodel.models.tiny_cnn import TinyCNN


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_loaders(root: str, batch_size: int, num_workers: int) -> tuple[DataLoader, DataLoader]:
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

    train_ds = datasets.CIFAR10(root=root, train=True, download=False, transform=train_tf)
    test_ds = datasets.CIFAR10(root=root, train=False, download=False, transform=test_tf)

    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=max(256, batch_size),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, test_loader


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(images)
        loss = F.cross_entropy(logits, labels)

        bs = images.size(0)
        total += bs
        total_loss += loss.item() * bs
        correct += (logits.argmax(dim=1) == labels).sum().item()

    return (total_loss / max(1, total)), (correct / max(1, total))


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    use_amp: bool,
) -> tuple[float, float, int]:
    model.train()
    amp_enabled = bool(use_amp and device.type == "cuda")
    scaler = torch.amp.GradScaler(device=device.type, enabled=amp_enabled)

    total_loss = 0.0
    correct = 0
    total = 0
    steps = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
            logits = model(images)
            loss = F.cross_entropy(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        bs = images.size(0)
        total += bs
        total_loss += loss.item() * bs
        correct += (logits.argmax(dim=1) == labels).sum().item()
        steps += 1

    return (total_loss / max(1, total)), (correct / max(1, total)), steps


def _resolve_checkpoint_path(resume: str | None) -> Path | None:
    if resume is None:
        return None
    p = Path(resume)
    if p.is_dir():
        # Accept a step directory like .../checkpoints/step_00000042/
        return p / "model.pt"
    return p


def save_checkpoint(
    *,
    out_dir: Path,
    epoch: int,
    global_step: int,
    best_acc: float,
    best_step: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> Path:
    """Save a checkpoint under models/<exp>/checkpoints/step_<N>/model.pt.

    Returns the path to the written checkpoint file.
    """
    step_dir = out_dir / "checkpoints" / f"step_{global_step:08d}"
    step_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = step_dir / "model.pt"
    torch.save(
        {
            "epoch": epoch,
            "global_step": global_step,
            "best_acc": best_acc,
            "best_step": best_step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        ckpt_path,
    )

    # Convenience pointers for resuming/inference.
    last_path = out_dir / "last" / "model.pt"
    last_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(torch.load(ckpt_path, map_location="cpu"), last_path)

    return ckpt_path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=str, default="data/raw")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--num-workers", type=int, default=0)  # start with 0 on Windows
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--exp-name", type=str, default="tinycnn_cifar10")
    p.add_argument("--resume", type=str, default=None, help="Path to a checkpoint file or step directory to resume from.")
    p.add_argument("--resume-latest", action="store_true", help="Resume from models/<exp-name>/last/model.pt if present.")
    args = p.parse_args()

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    if device.type == "cuda":
        print(f"gpu={torch.cuda.get_device_name(0)}")
        print(f"cuda_version={torch.version.cuda}")
    print(f"torch_version={torch.__version__}")

    train_loader, test_loader = make_loaders(args.data_root, args.batch_size, args.num_workers)

    model = TinyCNN(num_classes=10).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    out_dir = Path("models") / args.exp_name
    best_acc = 0.0
    best_step = 0
    global_step = 0
    start_epoch = 1

    resume_path = None
    if args.resume_latest:
        resume_path = out_dir / "last" / "model.pt"
    else:
        resume_path = _resolve_checkpoint_path(args.resume)

    if resume_path is not None and resume_path.exists():
        ckpt = torch.load(resume_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        global_step = int(ckpt.get("global_step", 0))
        best_acc = float(ckpt.get("best_acc", 0.0))
        best_step = int(ckpt.get("best_step", 0))
        print(f"resumed_from={resume_path} start_epoch={start_epoch} global_step={global_step} best_acc={best_acc:.4f}")

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()

        train_loss, train_acc, steps = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            use_amp=not args.no_amp,
        )
        global_step += steps
        test_loss, test_acc = evaluate(model, test_loader, device)

        dt = time.time() - t0

        print(
            f"epoch {epoch:03d}/{args.epochs:03d}  "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f}  "
            f"test_loss={test_loss:.4f} test_acc={test_acc:.4f}  "
            f"step={global_step}  time={dt:.1f}s"
        )

        if test_acc > best_acc:
            best_acc = test_acc
            best_step = global_step
            best_path = out_dir / "best" / "model.pt"
            best_path.parent.mkdir(parents=True, exist_ok=True)

        ckpt_path = save_checkpoint(
            out_dir=out_dir,
            epoch=epoch,
            global_step=global_step,
            best_acc=best_acc,
            best_step=best_step,
            model=model,
            optimizer=optimizer,
        )

        # Update best pointer if needed (write the most recent checkpoint content).
        if best_step == global_step:
            torch.save(torch.load(ckpt_path, map_location="cpu"), best_path)

    print(f"best_test_acc={best_acc:.4f}")


if __name__ == "__main__":
    main()
