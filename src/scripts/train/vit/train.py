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
from models.VITs import MicroViT, NanoViT, MiniViT, StandardViT, KiloViT


# -------------------------
# Registries (extend later)
# -------------------------

DatasetFactory = Callable[[argparse.Namespace], tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]]
ModelFactory = Callable[[], torch.nn.Module]

DATASET_SPECS: dict[str, dict[str, int]] = {
    "cifar10": {"num_classes": 10, "image_size": 32},
    "food101": {"num_classes": 101, "image_size": 224},
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
            image_size=int(args.image_size or 224),
        )
    ),
}

MODELS: dict[str, type[torch.nn.Module]] = {
    "nano_vit": NanoViT,
    "micro_vit": MicroViT,
    "mini_vit": MiniViT,
    "standard_vit": StandardViT,
    "kilo_vit": KiloViT,
}


def resolve_model_factory(model: str, model_kwargs: dict) -> ModelFactory:
    """
    Resolve a model spec to a factory returning nn.Module.

    Supported:
    - registry name: "micro_vit"
    - dotted path:   "some.module:ClassName" (must subclass torch.nn.Module)
    """
    if model in MODELS:
        cls = MODELS[model]
        return lambda: _instantiate_model(cls, **model_kwargs)

    if ":" in model:
        module_name, class_name = model.split(":", 1)
        mod = importlib.import_module(module_name)
        cls = getattr(mod, class_name)
        if not isinstance(cls, type) or not issubclass(cls, torch.nn.Module):
            raise SystemExit(f"--model {model} is not a torch.nn.Module subclass")
        return lambda: _instantiate_model(cls, **model_kwargs)

    raise SystemExit(f"Unknown --model '{model}'. Options: {', '.join(sorted(MODELS))} or 'module:ClassName'.")


def _instantiate_model(cls: type[torch.nn.Module], **kwargs) -> torch.nn.Module:
    """
    Instantiate a model class, passing through supported kwargs.
    Convention: pass num_classes=10 if supported.
    """
    sig = inspect.signature(cls.__init__)
    supported = {k: v for k, v in kwargs.items() if (k in sig.parameters and v is not None)}
    if "num_classes" in sig.parameters and "num_classes" not in supported:
        supported["num_classes"] = 10
    return cls(**supported)  # type: ignore[call-arg]


# -------------------------
# ViT-leaning utilities
# -------------------------


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_adamw_param_groups(model: torch.nn.Module, weight_decay: float) -> list[dict]:
    """
    Create AdamW param groups with "no decay" for:
    - biases
    - 1D params (LayerNorm / BatchNorm weights, etc.)
    - common ViT tokens/embeddings (pos_embed, cls_token)
    """
    decay: list[torch.nn.Parameter] = []
    no_decay: list[torch.nn.Parameter] = []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue

        is_bias = name.endswith(".bias")
        is_1d = p.ndim == 1
        is_vit_token = ("pos_embed" in name) or ("cls_token" in name)

        if is_bias or is_1d or is_vit_token:
            no_decay.append(p)
        else:
            decay.append(p)

    return [
        {"params": decay, "weight_decay": float(weight_decay)},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def _next_batch(train_iter, train_loader):
    try:
        return next(train_iter), train_iter
    except StopIteration:
        train_iter = iter(train_loader)
        return next(train_iter), train_iter


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


def _rand_bbox(size: tuple[int, int, int, int], lam: float) -> tuple[int, int, int, int]:
    # size: (B, C, H, W)
    h = size[2]
    w = size[3]
    cut_rat = float((1.0 - lam) ** 0.5)
    cut_w = int(w * cut_rat)
    cut_h = int(h * cut_rat)

    cx = int(torch.randint(0, w, (1,)).item())
    cy = int(torch.randint(0, h, (1,)).item())

    x1 = max(cx - cut_w // 2, 0)
    y1 = max(cy - cut_h // 2, 0)
    x2 = min(cx + cut_w // 2, w)
    y2 = min(cy + cut_h // 2, h)
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
    cfg: "TrainConfig",
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
    if best_acc == ckpt["best_acc"]:
        torch.save(ckpt, out_dir / "best.pt")


@dataclass(frozen=True)
class TrainConfig:
    dataset: str = "cifar10"
    model: str = "micro_vit"
    exp_name: str | None = None

    # Frequency
    save_freq: int = 10_000
    eval_freq: int = 1_000
    log_freq: int = 200

    # Run length
    steps: int = 100_000
    batch_size: int = 128
    num_workers: int = 0
    seed: int = 1337
    data_root: str = "data/raw"
    amp: bool = True

    # ViT-friendly optimizer defaults
    lr: float = 3e-4
    min_lr: float = 1e-6
    weight_decay: float = 0.05
    beta1: float = 0.9
    beta2: float = 0.999
    adam_eps: float = 1e-8

    # Schedule + regularization knobs
    warmup_steps: int = 5_000
    label_smoothing: float = 0.1
    mixup_alpha: float = 0.2
    cutmix_alpha: float = 1.0
    ema_decay: float = 0.9999
    grad_clip: float = 1.0  # 0 disables
    # Turn off MixUp/CutMix for the last N steps (often helps final top-1)
    mix_off_steps: int = 0
    # If True, evaluate both raw model and EMA (if EMA is enabled)
    eval_both: bool = False


def main() -> None:
    p = argparse.ArgumentParser()

    # Common
    p.add_argument("--dataset", type=str, default="cifar10", choices=sorted(DATASETS))
    p.add_argument("--model", type=str, default="micro_vit", help="Registry name (e.g. micro_vit) or 'module:ClassName'")
    p.add_argument("--exp-name", type=str, default=None, help="Default: <model>_<dataset>")
    p.add_argument("--data-root", type=str, default="data/raw")
    p.add_argument("--download", action="store_true", help="Download dataset if missing")
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--seed", type=int, default=1337)

    # Training loop
    p.add_argument("--steps", type=int, default=100_000)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--log-freq", type=int, default=200)
    p.add_argument("--eval-freq", type=int, default=1_000)
    p.add_argument("--save-freq", type=int, default=10_000)
    p.add_argument("--resume", type=str, default=None, help="Path to a checkpoint .pt (e.g. models/<exp>/last.pt)")
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--compile", action="store_true", help="Use torch.compile if available (PyTorch 2.x).")

    # Model hyperparams (passed to model __init__ if supported)
    p.add_argument("--image-size", type=int, default=None)
    p.add_argument("--patch-size", type=int, default=None)
    p.add_argument("--dim", type=int, default=None)
    p.add_argument("--depth", type=int, default=None)
    p.add_argument("--num-heads", type=int, default=None)
    p.add_argument("--mlp-ratio", type=float, default=None)
    p.add_argument("--drop", type=float, default=None)
    p.add_argument("--attn-drop", type=float, default=None)
    p.add_argument("--drop-path-rate", type=float, default=None)
    p.add_argument("--pool", type=str, default=None, help="If supported: 'cls' or 'mean'")

    # AdamW
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--min-lr", type=float, default=1e-6)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--beta1", type=float, default=0.9)
    p.add_argument("--beta2", type=float, default=0.999)
    p.add_argument("--adam-eps", type=float, default=1e-8)

    # Schedule + recipe
    p.add_argument("--warmup-steps", type=int, default=5_000)
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--mixup-alpha", type=float, default=0.2)
    p.add_argument("--cutmix-alpha", type=float, default=1.0)
    p.add_argument("--ema-decay", type=float, default=0.9999)
    p.add_argument("--grad-clip", type=float, default=1.0, help="Max grad norm; 0 disables.")
    p.add_argument("--mix-off-steps", type=int, default=0, help="Disable MixUp/CutMix for the last N steps.")
    p.add_argument(
        "--eval-both",
        action="store_true",
        help="Evaluate both raw model and EMA model (if EMA enabled).",
    )

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

    # Nice default for conv-heavy augmentations and patch embed convs
    torch.backends.cudnn.benchmark = (device.type == "cuda")

    # Environment / hardware info
    print(f"torch={torch.__version__}")
    if device.type == "cuda":
        print(f"device=cuda gpu={torch.cuda.get_device_name(0)} cuda={torch.version.cuda}")
    else:
        cpu = platform.processor() or platform.machine() or "unknown"
        print(f"device=cpu cpu={cpu} cores={os.cpu_count()}")

    set_seed(int(args.seed))

    cfg = TrainConfig(
        dataset=str(args.dataset),
        model=str(args.model),
        exp_name=str(exp_name),
        save_freq=int(args.save_freq),
        eval_freq=int(args.eval_freq),
        log_freq=int(args.log_freq),
        steps=int(args.steps),
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        seed=int(args.seed),
        data_root=str(args.data_root),
        amp=not bool(args.no_amp),
        lr=float(args.lr),
        min_lr=float(args.min_lr),
        weight_decay=float(args.weight_decay),
        beta1=float(args.beta1),
        beta2=float(args.beta2),
        adam_eps=float(args.adam_eps),
        warmup_steps=int(args.warmup_steps),
        label_smoothing=float(args.label_smoothing),
        mixup_alpha=float(args.mixup_alpha),
        cutmix_alpha=float(args.cutmix_alpha),
        ema_decay=float(args.ema_decay),
        grad_clip=float(args.grad_clip),
        mix_off_steps=int(args.mix_off_steps),
        eval_both=bool(args.eval_both),
    )

    print(
        "config "
        f"dataset={cfg.dataset} model={cfg.model} exp={cfg.exp_name} "
        f"steps={cfg.steps} batch_size={cfg.batch_size} "
        f"lr={cfg.lr} min_lr={cfg.min_lr} weight_decay={cfg.weight_decay} warmup_steps={cfg.warmup_steps} "
        f"label_smoothing={cfg.label_smoothing} mixup_alpha={cfg.mixup_alpha} cutmix_alpha={cfg.cutmix_alpha} "
        f"ema_decay={cfg.ema_decay} grad_clip={cfg.grad_clip} "
        f"num_workers={cfg.num_workers} seed={cfg.seed} amp={cfg.amp} data_root={cfg.data_root}"
    )

    spec = DATASET_SPECS[args.dataset]
    model_kwargs = {
        "image_size": getattr(args, "image_size", None),
        "patch_size": getattr(args, "patch_size", None),
        "dim": getattr(args, "dim", None),
        "depth": getattr(args, "depth", None),
        "num_heads": getattr(args, "num_heads", None),
        "mlp_ratio": getattr(args, "mlp_ratio", None),
        "drop": getattr(args, "drop", None),
        "attn_drop": getattr(args, "attn_drop", None),
        "drop_path_rate": getattr(args, "drop_path_rate", None),
        "pool": getattr(args, "pool", None),
    }
    # Ensure the head matches dataset labels and (by default) model pos_embed matches input size.
    model_kwargs["num_classes"] = int(spec["num_classes"])
    if model_kwargs.get("image_size") is None:
        model_kwargs["image_size"] = int(spec["image_size"])

    train_loader, test_loader = DATASETS[args.dataset](args)
    model_factory = resolve_model_factory(args.model, model_kwargs=model_kwargs)
    model = model_factory().to(device)

    if bool(args.compile) and hasattr(torch, "compile"):
        model = torch.compile(model)  # type: ignore[attr-defined]

    # EMA (optional but recommended for ViTs)
    ema_model: torch.nn.Module | None = None
    if float(cfg.ema_decay) > 0.0:
        ema_model = deepcopy(model).to(device)
        ema_model.eval()
        for p_ema in ema_model.parameters():
            p_ema.requires_grad_(False)

    # AdamW with ViT-style no-decay param groups
    param_groups = build_adamw_param_groups(model, weight_decay=cfg.weight_decay)
    optimizer = torch.optim.AdamW(
        param_groups,
        lr=cfg.lr,
        betas=(cfg.beta1, cfg.beta2),
        eps=cfg.adam_eps,
    )

    total_steps = int(cfg.steps)
    warmup_steps = int(max(0, cfg.warmup_steps))
    base_lr = float(cfg.lr)
    min_lr = float(cfg.min_lr)

    def lr_lambda(step: int) -> float:
        # Scheduler returns a multiplier for base LR. We implement:
        # - linear warmup to base_lr
        # - cosine decay from base_lr to min_lr
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        # cosine over remaining steps
        t = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        t = min(max(t, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(t * math.pi))
        # map [0..1] to [min_lr/base_lr .. 1]
        min_ratio = (min_lr / base_lr) if base_lr > 0 else 0.0
        return min_ratio + (1.0 - min_ratio) * cosine

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    out_dir = Path("models") / exp_name
    global_step = 0
    best_acc = 0.0

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
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

    amp_enabled = bool(cfg.amp and device.type == "cuda")
    scaler = torch.amp.GradScaler(device=device.type, enabled=amp_enabled)
    train_iter = iter(train_loader)

    # Running stats for logging (over last log window)
    win_loss_sum = 0.0
    win_correct = 0.0
    win_total = 0
    win_steps = 0
    t_window0 = time.time()

    while global_step < total_steps:
        (x, y), train_iter = _next_batch(train_iter, train_loader)
        global_step += 1

        model.train()
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        # MixUp / CutMix (optional)
        mixup_alpha = float(cfg.mixup_alpha)
        cutmix_alpha = float(cfg.cutmix_alpha)
        # Optional: turn off MixUp/CutMix for the last N steps.
        mix_off_steps = int(cfg.mix_off_steps)
        mixing_allowed = (mix_off_steps <= 0) or (global_step <= (total_steps - mix_off_steps))
        do_mix = mixing_allowed and ((mixup_alpha > 0.0) or (cutmix_alpha > 0.0))
        y_a = y
        y_b = y
        lam = 1.0
        if do_mix:
            use_cutmix = (cutmix_alpha > 0.0) and ((mixup_alpha <= 0.0) or (torch.rand(()) < 0.5))
            alpha = cutmix_alpha if use_cutmix else mixup_alpha
            lam = float(torch.distributions.Beta(alpha, alpha).sample(()).item())
            perm = torch.randperm(x.size(0), device=x.device)
            y_b = y[perm]
            if use_cutmix:
                x2 = x[perm]
                x1, y1, x2b, y2 = _rand_bbox(tuple(x.size()), lam)
                x[:, :, y1:y2, x1:x2b] = x2[:, :, y1:y2, x1:x2b]
                lam = 1.0 - ((x2b - x1) * (y2 - y1) / float(x.size(-1) * x.size(-2)))
            else:
                x = x * lam + x[perm] * (1.0 - lam)

        with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
            logits = model(x)
            if do_mix:
                loss_a = F.cross_entropy(
                    logits,
                    y_a,
                    label_smoothing=float(cfg.label_smoothing),
                    reduction="none",
                )
                loss_b = F.cross_entropy(
                    logits,
                    y_b,
                    label_smoothing=float(cfg.label_smoothing),
                    reduction="none",
                )
                loss = _mix_targets(loss_a, loss_b, lam).mean()
            else:
                loss = F.cross_entropy(logits, y, label_smoothing=float(cfg.label_smoothing))

        scaler.scale(loss).backward()

        # Optional grad clipping (after unscale, before step)
        if float(cfg.grad_clip) and float(cfg.grad_clip) > 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(cfg.grad_clip))

        scaler.step(optimizer)
        scaler.update()

        # EMA update (after optimizer step)
        if ema_model is not None:
            decay = float(cfg.ema_decay)
            with torch.no_grad():
                msd = model.state_dict()
                esd = ema_model.state_dict()
                for k, v in esd.items():
                    mv = msd[k]
                    if torch.is_floating_point(v) and torch.is_floating_point(mv):
                        v.mul_(decay).add_(mv, alpha=(1.0 - decay))
                    else:
                        v.copy_(mv)

        scheduler.step()

        bs = x.size(0)
        win_total += bs
        win_loss_sum += float(loss.item()) * bs
        pred = logits.argmax(1)
        if do_mix:
            # "Soft"/expected accuracy under MixUp/CutMix:
            # correct = lam * 1[pred==y_a] + (1-lam) * 1[pred==y_b]
            win_correct += float((pred == y_a).float().sum().item()) * lam + float((pred == y_b).float().sum().item()) * (
                1.0 - lam
            )
        else:
            win_correct += float((pred == y).float().sum().item())
        win_steps += 1

        if cfg.log_freq > 0 and (global_step % cfg.log_freq == 0):
            dt = time.time() - t_window0
            train_loss = win_loss_sum / max(1, win_total)
            train_acc = win_correct / max(1, win_total)
            steps_per_s = win_steps / max(1e-9, dt)
            lr_now = optimizer.param_groups[0]["lr"]
            print(
                f"step {global_step}/{total_steps}  "
                f"lr={lr_now:.6f}  "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.4f}  "
                f"steps/s={steps_per_s:.1f}"
            )
            win_loss_sum = 0.0
            win_correct = 0.0
            win_total = 0
            win_steps = 0
            t_window0 = time.time()

        if cfg.eval_freq > 0 and (global_step % cfg.eval_freq == 0):
            t0 = time.time()
            ema_loss = None
            ema_acc = None
            if ema_model is not None:
                ema_loss, ema_acc = evaluate(ema_model, test_loader, device)

            raw_loss = None
            raw_acc = None
            if bool(cfg.eval_both) or (ema_model is None):
                raw_loss, raw_acc = evaluate(model, test_loader, device)

            dt = time.time() - t0

            tracked_acc = ema_acc if ema_acc is not None else raw_acc
            if tracked_acc is not None and tracked_acc > best_acc:
                best_acc = tracked_acc

            parts: list[str] = [f"eval step {global_step}"]
            if raw_acc is not None and raw_loss is not None:
                parts.append(f"raw_loss={raw_loss:.4f} raw_acc={raw_acc:.4f}")
            if ema_acc is not None and ema_loss is not None:
                parts.append(f"ema_loss={ema_loss:.4f} ema_acc={ema_acc:.4f}")
            parts.append(f"best_acc={best_acc:.4f} time={dt:.1f}s")
            print("  ".join(parts))

        if cfg.save_freq > 0 and (global_step % cfg.save_freq == 0):
            save_checkpoint(
                out_dir=out_dir,
                global_step=global_step,
                best_acc=best_acc,
                model=model,
                ema_model=ema_model,
                optimizer=optimizer,
                cfg=cfg,
            )

    # Final eval + checkpoint
    t0 = time.time()
    # Report both raw and EMA (if present) at the end.
    final_raw_loss, final_raw_acc = evaluate(model, test_loader, device)
    final_ema_loss, final_ema_acc = (None, None)
    if ema_model is not None:
        final_ema_loss, final_ema_acc = evaluate(ema_model, test_loader, device)
    dt = time.time() - t0
    final_tracked_acc = final_ema_acc if final_ema_acc is not None else final_raw_acc
    if final_tracked_acc > best_acc:
        best_acc = final_tracked_acc

    lr_now = optimizer.param_groups[0]["lr"]
    parts: list[str] = [
        f"model={cfg.model}",
        f"final eval step {global_step}",
        f"lr={lr_now:.6f}",
        f"raw_loss={final_raw_loss:.4f} raw_acc={final_raw_acc:.4f}",
    ]
    if final_ema_acc is not None and final_ema_loss is not None:
        parts.append(f"ema_loss={final_ema_loss:.4f} ema_acc={final_ema_acc:.4f}")
    parts.append(f"best_acc={best_acc:.4f} time={dt:.1f}s")
    print("  ".join(parts))

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
