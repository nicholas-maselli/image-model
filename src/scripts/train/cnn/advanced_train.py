from __future__ import annotations

import argparse
import importlib
import inspect
import json
import os
import platform
import random
import time
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import math
import torch
import torch.nn.functional as F

from data import Cifar10DataConfig, Food101DataConfig, make_cifar10_loaders, make_food101_loaders
from models import MicroCNN, MilliCNN, StandardCNN, KiloCNN, TestCandidate0


# -------------------------
# Registries (extend later)
# -------------------------

DatasetFactory = Callable[[argparse.Namespace], tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]]
ModelFactory = Callable[[], torch.nn.Module]

DATASET_SPECS: dict[str, dict[str, int]] = {
    "cifar10": {"num_classes": 10},
    "food101": {"num_classes": 101},
}

DATASETS: dict[str, DatasetFactory] = {
    "cifar10": lambda args: make_cifar10_loaders(
        Cifar10DataConfig(
            data_root=args.data_root,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            download=args.download,
        )
    ),
    "food101": lambda args: make_food101_loaders(
        Food101DataConfig(
            data_root=args.data_root,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            download=args.download,
            image_size=224,
        )
    ),
}

MODELS: dict[str, type[torch.nn.Module]] = {
    "micro": MicroCNN,
    "milli": MilliCNN,
    "standard": StandardCNN,
    "kilo": KiloCNN,
    "test_candidate_cuda0": TestCandidate0
}


def resolve_model_factory(model: str, *, num_classes: int) -> ModelFactory:
    """
    Resolve a model spec to a factory returning nn.Module.

    Supported:
    - registry name: "micro"
    - dotted path:   "some.module:ClassName" (must subclass torch.nn.Module)
    """
    if model in MODELS:
        cls = MODELS[model]
        return lambda: _instantiate_model(cls, num_classes=num_classes)

    if ":" in model:
        module_name, class_name = model.split(":", 1)
        mod = importlib.import_module(module_name)
        cls = getattr(mod, class_name)
        if not isinstance(cls, type) or not issubclass(cls, torch.nn.Module):
            raise SystemExit(f"--model {model} is not a torch.nn.Module subclass")
        return lambda: _instantiate_model(cls, num_classes=num_classes)

    raise SystemExit(f"Unknown --model '{model}'. Options: {', '.join(sorted(MODELS))} or 'module:ClassName'.")


def _instantiate_model(cls: type[torch.nn.Module], *, num_classes: int) -> torch.nn.Module:
    # Convention: pass num_classes if supported; otherwise call with no args.
    sig = inspect.signature(cls.__init__)
    if "num_classes" in sig.parameters:
        return cls(num_classes=num_classes)  # type: ignore[call-arg]
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
    lr: float = 0.1
    weight_decay: float = 5e-4
    num_workers: int = 0
    seed: int = 1337
    amp: bool = False
    data_root: str = "data/raw"
    label_smoothing: float = 0.0

    # Optimizer Parameters
    opt: str = "sgd"
    momentum: float = 0.9
    nesterov: bool = False
    sched: str = "cosine"
    warmup_steps: int = 0

    # Regularization / recipe knobs (optional)
    mixup_alpha: float = 0.0
    cutmix_alpha: float = 0.0
    ema_decay: float = 0.0


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


def _rand_bbox(size: tuple[int, int, int, int], lam: float) -> tuple[int, int, int, int]:
    # size: (B, C, H, W)
    H = size[2]
    W = size[3]
    cut_rat = float((1.0 - lam) ** 0.5)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    cx = int(torch.randint(0, W, (1,)).item())
    cy = int(torch.randint(0, H, (1,)).item())

    x1 = max(cx - cut_w // 2, 0)
    y1 = max(cy - cut_h // 2, 0)
    x2 = min(cx + cut_w // 2, W)
    y2 = min(cy + cut_h // 2, H)
    return x1, y1, x2, y2


def _mix_targets(loss_a: torch.Tensor, loss_b: torch.Tensor, lam: float) -> torch.Tensor:
    return loss_a * lam + loss_b * (1.0 - lam)


def save_checkpoint(
    *,
    out_dir: Path,
    global_step: int,
    best_acc: float,
    model: torch.nn.Module,
    ema_model: torch.nn.Module | None,
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
        "ema_model_state_dict": (ema_model.state_dict() if ema_model is not None else None),
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
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument(
        "--label-smoothing",
        type=float,
        default=0.0,
        help="Label smoothing for cross entropy. Typical: 0.0 (off), 0.05, 0.1.",
    )

    p.add_argument("--data-root", type=str, default="data/raw")
    p.add_argument("--download", action="store_true", help="Download dataset if missing")

    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--resume", type=str, default=None, help="Path to a checkpoint .pt (e.g. models/<exp>/last.pt)")
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])

    p.add_argument("--opt", type=str, default="sgd", choices=["sgd", "adamw"])
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--nesterov", action="store_true")
    p.add_argument("--sched", type=str, default="cosine", choices=["none", "cosine", "multistep"])
    p.add_argument("--warmup-steps", type=int, default=0, help="Linear LR warmup steps (0 disables).")

    # Optional 'strong recipe' knobs (all default off)
    p.add_argument("--mixup-alpha", type=float, default=0.0, help="MixUp alpha; 0 disables (typical: 0.2).")
    p.add_argument("--cutmix-alpha", type=float, default=0.0, help="CutMix alpha; 0 disables (typical: 1.0).")
    p.add_argument("--ema-decay", type=float, default=0.0, help="EMA decay for eval (0 disables; typical: 0.999-0.9999).")

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
        label_smoothing=float(args.label_smoothing),
        opt=args.opt,
        momentum=args.momentum,
        nesterov=args.nesterov,
        sched=args.sched,
        warmup_steps=int(args.warmup_steps),
        mixup_alpha=float(args.mixup_alpha),
        cutmix_alpha=float(args.cutmix_alpha),
        ema_decay=float(args.ema_decay),
    )

    print(
        "config "
        f"dataset={cfg.dataset} model={cfg.model} exp={cfg.exp_name} "
        f"steps={cfg.steps} batch_size={cfg.batch_size} "
        f"lr={cfg.lr} weight_decay={cfg.weight_decay} "
        f"save_freq={cfg.save_freq} eval_freq={cfg.eval_freq} log_freq={cfg.log_freq} "
        f"num_workers={cfg.num_workers} seed={cfg.seed} amp={cfg.amp} data_root={cfg.data_root} "
        f"label_smoothing={cfg.label_smoothing} warmup_steps={cfg.warmup_steps} "
        f"mixup_alpha={cfg.mixup_alpha} cutmix_alpha={cfg.cutmix_alpha} ema_decay={cfg.ema_decay}"
    )

    train_loader, test_loader = DATASETS[args.dataset](args)
    num_classes = int(DATASET_SPECS[args.dataset]["num_classes"])
    model_factory = resolve_model_factory(args.model, num_classes=num_classes)
    model = model_factory().to(device)
    ema_model: torch.nn.Module | None = None
    if float(args.ema_decay) > 0.0:
        ema_model = deepcopy(model).to(device)
        ema_model.eval()
        for p in ema_model.parameters():
            p.requires_grad_(False)
    if args.opt == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            nesterov=args.nesterov,
        )
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    steps_per_epoch = len(train_loader)
    total_steps = args.steps

    if args.sched == "cosine":
        warmup_steps = int(getattr(args, "warmup_steps", 0))
        base_lr = float(args.lr)

        def lr_lambda(step: int) -> float:
            # step is 0-based in LambdaLR; we are calling scheduler.step() once per optimizer step.
            if warmup_steps > 0 and step < warmup_steps:
                return float(step + 1) / float(max(1, warmup_steps))
            # cosine from base_lr down to ~0 over the remaining steps
            t = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            t = min(max(t, 0.0), 1.0)
            return 0.5 * (1.0 + math.cos(t * math.pi))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    elif args.sched == "multistep":
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[int(total_steps * 0.6), int(total_steps * 0.8)],
            gamma=0.2,
        )
    else:
        scheduler = None


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
        if ema_model is not None:
            ema_sd = ckpt.get("ema_model_state_dict")
            if isinstance(ema_sd, dict):
                ema_model.load_state_dict(ema_sd)
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

        # Optional MixUp / CutMix. (If both enabled, prefer CutMix 50% of the time.)
        mixup_alpha = float(getattr(args, "mixup_alpha", 0.0))
        cutmix_alpha = float(getattr(args, "cutmix_alpha", 0.0))
        do_mix = (mixup_alpha > 0.0) or (cutmix_alpha > 0.0)
        y_a = y
        y_b = y
        lam = 1.0
        if do_mix:
            use_cutmix = (cutmix_alpha > 0.0) and ((mixup_alpha <= 0.0) or (torch.rand(()) < 0.5))
            alpha = cutmix_alpha if use_cutmix else mixup_alpha
            # Beta(alpha, alpha)
            lam = float(torch.distributions.Beta(alpha, alpha).sample(()).item())
            perm = torch.randperm(x.size(0), device=x.device)
            y_b = y[perm]
            if use_cutmix:
                x2 = x[perm]
                x1, y1, x2b, y2 = _rand_bbox(tuple(x.size()), lam)
                x[:, :, y1:y2, x1:x2b] = x2[:, :, y1:y2, x1:x2b]
                # adjust lambda to exact pixel ratio
                lam = 1.0 - ((x2b - x1) * (y2 - y1) / float(x.size(-1) * x.size(-2)))
            else:
                x = x * lam + x[perm] * (1.0 - lam)

        with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
            logits = model(x)
            if do_mix:
                loss_a = F.cross_entropy(logits, y_a, label_smoothing=float(args.label_smoothing), reduction="none")
                loss_b = F.cross_entropy(logits, y_b, label_smoothing=float(args.label_smoothing), reduction="none")
                loss = _mix_targets(loss_a, loss_b, lam).mean()
            else:
                loss = F.cross_entropy(logits, y, label_smoothing=float(args.label_smoothing))

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # EMA update (after optimizer step)
        if ema_model is not None:
            decay = float(args.ema_decay)
            with torch.no_grad():
                msd = model.state_dict()
                esd = ema_model.state_dict()
                for k, v in esd.items():
                    mv = msd[k]
                    if torch.is_floating_point(v) and torch.is_floating_point(mv):
                        v.mul_(decay).add_(mv, alpha=(1.0 - decay))
                    else:
                        v.copy_(mv)

        if scheduler is not None:
            scheduler.step()

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
            eval_model = ema_model if ema_model is not None else model
            test_loss, test_acc = evaluate(eval_model, test_loader, device)
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
                ema_model=ema_model,
                optimizer=optimizer,
                cfg=cfg,
            )

    # Final eval + checkpoint (so you always get end-of-run numbers and a matching checkpoint)
    t0 = time.time()
    final_test_loss, final_test_acc = evaluate(model, test_loader, device)
    dt = time.time() - t0
    if final_test_acc > best_acc:
        best_acc = final_test_acc

    lr = optimizer.param_groups[0]['lr']
    print(
        f"model={cfg.model} "
        f"final eval step {global_step}  "
        f"lr={lr:.6f} "
        f"test_loss={final_test_loss:.4f} test_acc={final_test_acc:.4f}  "
        f"best_acc={best_acc:.4f}  time={dt:.1f}s"
    )

    save_checkpoint(
        out_dir=out_dir,
        global_step=global_step,
        best_acc=best_acc,
        model=model,
        ema_model=ema_model,
        optimizer=optimizer,
        cfg=cfg,
    )

    print(f"best_test_acc={best_acc:.4f}")


if __name__ == "__main__":
    main()
