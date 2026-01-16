from __future__ import annotations

import argparse
import importlib
import inspect
import json
import os
import platform
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as F

from data import Cifar10DataConfig, make_cifar10_loaders
from models import MicroCNN, MilliCNN, StandardCNN, KiloCNN, TestCandidate


# -------------------------
# Registries (extend later)
# -------------------------

DatasetFactory = Callable[[argparse.Namespace], tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]]
ModelFactory = Callable[[], torch.nn.Module]

DATASETS: dict[str, DatasetFactory] = {
    "cifar10": lambda args: make_cifar10_loaders(
        Cifar10DataConfig(
            data_root=args.data_root,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            download=args.download,
        )
    ),
}

MODELS: dict[str, type[torch.nn.Module]] = {
    "micro": MicroCNN,
    "milli": MilliCNN,
    "standard": StandardCNN,
    "kilo": KiloCNN,
    "test_candidate": TestCandidate,
}


def resolve_model_factory(model: str) -> ModelFactory:
    """
    Resolve a model spec to a factory returning nn.Module.

    Supported:
    - registry name: "micro"
    - dotted path:   "some.module:ClassName" (must subclass torch.nn.Module)
    """
    if model in MODELS:
        cls = MODELS[model]
        return lambda: _instantiate_model(cls)

    if ":" in model:
        module_name, class_name = model.split(":", 1)
        mod = importlib.import_module(module_name)
        cls = getattr(mod, class_name)
        if not isinstance(cls, type) or not issubclass(cls, torch.nn.Module):
            raise SystemExit(f"--model {model} is not a torch.nn.Module subclass")
        return lambda: _instantiate_model(cls)

    raise SystemExit(f"Unknown --model '{model}'. Options: {', '.join(sorted(MODELS))} or 'module:ClassName'.")


def _instantiate_model(cls: type[torch.nn.Module]) -> torch.nn.Module:
    # Convention: pass num_classes=10 if supported; otherwise call with no args.
    sig = inspect.signature(cls.__init__)
    if "num_classes" in sig.parameters:
        return cls(num_classes=10)  # type: ignore[call-arg]
    return cls()  # type: ignore[call-arg]


# -------------------------
# Training utilities
# -------------------------

@dataclass(frozen=True)
class TrainConfig:
    dataset: str = "cifar10"
    model: str = "micro"
    exp_name: str | None = None

    # Checkpoint and Evaluation Frequency
    save_freq: int = 10_000
    eval_freq: int = 1_000
    log_freq: int = 200

    # Model Parameters
    steps: int = 100_000
    batch_size: int = 128
    lr: float = 3e-4
    weight_decay: float = 1e-4
    num_workers: int = 0
    seed: int = 1337
    amp: bool = False
    data_root: str = "data/raw"


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader, device: torch.device) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        loss = F.cross_entropy(logits, y)

        bs = x.size(0)
        total += bs
        total_loss += loss.item() * bs
        correct += (logits.argmax(1) == y).sum().item()

    return total_loss / max(1, total), correct / max(1, total)


def _next_batch(train_iter, train_loader):
    try:
        return next(train_iter), train_iter
    except StopIteration:
        train_iter = iter(train_loader)
        return next(train_iter), train_iter


def save_checkpoint(
    *,
    out_dir: Path,
    global_step: int,
    best_acc: float,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    cfg: TrainConfig,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")

    ckpt = {
        "global_step": global_step,
        "best_acc": best_acc,
        "model": cfg.model,
        "dataset": cfg.dataset,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": asdict(cfg),
    }

    ckpt_path = out_dir / "checkpoints" / f"step_{global_step}.pt"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, ckpt_path)

    torch.save(ckpt, out_dir / "last.pt")
    # Best pointer updates when best_acc improves (caller is responsible for updating best_acc).
    if best_acc == ckpt["best_acc"]:
        torch.save(ckpt, out_dir / "best.pt")


def main() -> None:
    p = argparse.ArgumentParser()

    p.add_argument("--dataset", type=str, default="cifar10", choices=sorted(DATASETS))
    p.add_argument("--model", type=str, default="micro", help="Registry name (e.g. micro) or 'module:ClassName'")
    p.add_argument("--exp-name", type=str, default=None, help="Default: <model>_<dataset>")

    p.add_argument("--steps", type=int, default=100_000, help="Total optimizer steps to run")
    p.add_argument("--save-freq", type=int, default=10_000, help="Checkpoint every N steps")
    p.add_argument("--eval-freq", type=int, default=1_000, help="Evaluate on test set every N steps")
    p.add_argument("--log-freq", type=int, default=100, help="Log running train stats every N steps")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=1337)

    p.add_argument("--data-root", type=str, default="data/raw")
    p.add_argument("--download", action="store_true", help="Download dataset if missing")

    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--resume", type=str, default=None, help="Path to a checkpoint .pt (e.g. models/<exp>/last.pt)")
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])

    args = p.parse_args()

    exp_name = args.exp_name or f"{args.model.replace(':', '_')}_{args.dataset}"

    if args.device == "cpu":
        device = torch.device("cpu")
    elif args.device == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("CUDA requested but torch.cuda.is_available() is False")
        device = torch.device("cuda")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Environment / hardware info (useful for reproducibility + debugging)
    print(f"torch={torch.__version__}")
    if device.type == "cuda":
        print(f"device=cuda gpu={torch.cuda.get_device_name(0)} cuda={torch.version.cuda}")
    else:
        cpu = platform.processor() or platform.machine() or "unknown"
        print(f"device=cpu cpu={cpu} cores={os.cpu_count()}")

    set_seed(args.seed)

    cfg = TrainConfig(
        dataset=args.dataset,
        model=args.model,
        exp_name=exp_name,
        save_freq=args.save_freq,
        eval_freq=args.eval_freq,
        log_freq=args.log_freq,
        steps=args.steps,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
        seed=args.seed,
        amp=not args.no_amp,
        data_root=args.data_root,
    )

    print(
        "config "
        f"dataset={cfg.dataset} model={cfg.model} exp={cfg.exp_name} "
        f"steps={cfg.steps} batch_size={cfg.batch_size} "
        f"lr={cfg.lr} weight_decay={cfg.weight_decay} "
        f"save_freq={cfg.save_freq} eval_freq={cfg.eval_freq} log_freq={cfg.log_freq} "
        f"num_workers={cfg.num_workers} seed={cfg.seed} amp={cfg.amp} data_root={cfg.data_root}"
    )

    train_loader, test_loader = DATASETS[args.dataset](args)
    model_factory = resolve_model_factory(args.model)
    model = model_factory().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    out_dir = Path("models") / exp_name
    global_step = 0
    best_acc = 0.0

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        # Helpful safety check
        ckpt_model = ckpt.get("model")
        ckpt_dataset = ckpt.get("dataset")
        if ckpt_model is not None and ckpt_model != args.model:
            raise SystemExit(f"Resume mismatch: ckpt.model={ckpt_model} but --model={args.model}")
        if ckpt_dataset is not None and ckpt_dataset != args.dataset:
            raise SystemExit(f"Resume mismatch: ckpt.dataset={ckpt_dataset} but --dataset={args.dataset}")

        model.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        global_step = int(ckpt.get("global_step", 0))
        best_acc = float(ckpt.get("best_acc", 0.0))
        print(f"resumed_from={args.resume} global_step={global_step} best_acc={best_acc:.4f}")

    print()

    amp_enabled = bool((not args.no_amp) and device.type == "cuda")
    scaler = torch.amp.GradScaler(device=device.type, enabled=amp_enabled)
    train_iter = iter(train_loader)

    # Running stats for logging (over last log window)
    win_loss_sum = 0.0
    win_correct = 0
    win_total = 0
    win_steps = 0
    t_window0 = time.time()

    while global_step < args.steps:
        (x, y), train_iter = _next_batch(train_iter, train_loader)
        global_step += 1

        model.train()
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
            logits = model(x)
            loss = F.cross_entropy(logits, y)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        bs = x.size(0)
        win_total += bs
        win_loss_sum += float(loss.item()) * bs
        win_correct += int((logits.argmax(1) == y).sum().item())
        win_steps += 1

        if args.log_freq > 0 and (global_step % args.log_freq == 0):
            dt = time.time() - t_window0
            train_loss = win_loss_sum / max(1, win_total)
            train_acc = win_correct / max(1, win_total)
            steps_per_s = win_steps / max(1e-9, dt)
            print(
                f"step {global_step}/{args.steps}  "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.4f}  "
                f"steps/s={steps_per_s:.1f}"
            )
            # reset window
            win_loss_sum = 0.0
            win_correct = 0
            win_total = 0
            win_steps = 0
            t_window0 = time.time()

        if args.eval_freq > 0 and (global_step % args.eval_freq == 0):
            t0 = time.time()
            test_loss, test_acc = evaluate(model, test_loader, device)
            dt = time.time() - t0
            if test_acc > best_acc:
                best_acc = test_acc
            print(
                f"eval step {global_step}  "
                f"test_loss={test_loss:.4f} test_acc={test_acc:.4f}  "
                f"best_acc={best_acc:.4f}  time={dt:.1f}s"
            )

        if args.save_freq > 0 and (global_step % args.save_freq == 0):
            save_checkpoint(
                out_dir=out_dir,
                global_step=global_step,
                best_acc=best_acc,
                model=model,
                optimizer=optimizer,
                cfg=cfg,
            )

    # Final eval + checkpoint (so you always get end-of-run numbers and a matching checkpoint)
    t0 = time.time()
    final_test_loss, final_test_acc = evaluate(model, test_loader, device)
    dt = time.time() - t0
    if final_test_acc > best_acc:
        best_acc = final_test_acc
    print(
        f"final eval step {global_step}  "
        f"test_loss={final_test_loss:.4f} test_acc={final_test_acc:.4f}  "
        f"best_acc={best_acc:.4f}  time={dt:.1f}s"
    )

    save_checkpoint(
        out_dir=out_dir,
        global_step=global_step,
        best_acc=best_acc,
        model=model,
        optimizer=optimizer,
        cfg=cfg,
    )

    print(f"best_test_acc={best_acc:.4f}")


if __name__ == "__main__":
    main()
