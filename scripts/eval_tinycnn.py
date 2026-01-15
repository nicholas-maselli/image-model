import argparse
import time
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from imagemodel.models.tiny_cnn import TinyCNN

# CIFAR-10 class order (matches torchvision CIFAR10 labels 0..9)
CLASSES = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)

# Your ground-truth labels (1-indexed filenames)
TRUE_LABELS = {
    1: "airplane",
    2: "airplane",
    3: "ship",
    4: "ship",
    5: "frog",
    6: "frog",
    7: "bird",
    8: "bird",
    9: "horse",
    10: "horse",
    11: "truck",
    12: "truck",
    13: "deer",
    14: "deer",
    15: "deer",
    16: "dog",
    17: "dog",
    18: "dog",
    19: "dog",
    20: "automobile",
    21: "automobile",
    22: "automobile",
    23: "cat",
    24: "cat",
    25: "cat",
}

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def load_model(ckpt_path: Path, device: torch.device) -> TinyCNN:
    model = TinyCNN(num_classes=10).to(device)
    ckpt = torch.load(str(ckpt_path), map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def format_topk(probs: torch.Tensor, k: int = 3) -> str:
    vals, idxs = torch.topk(probs, k=k)
    parts = []
    for p, i in zip(vals.tolist(), idxs.tolist()):
        parts.append(f"{CLASSES[i]}={p:.3f}")
    return "  ".join(parts)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--images-dir", type=str, default="data/test_images")
    p.add_argument("--ckpt", type=str, default="models/tinycnn_cifar10/best/model.pt")
    p.add_argument("--n", type=int, default=25, help="How many images test_1..test_n to evaluate")
    args = p.parse_args()

    images_dir = Path(args.images_dir)
    ckpt_path = Path(args.ckpt)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")
    if device.type == "cuda":
        print(f"gpu={torch.cuda.get_device_name(0)}  cuda={torch.version.cuda}")

    model = load_model(ckpt_path, device)

    tf = transforms.Compose(
        [
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )

    correct_top1 = 0
    correct_top3 = 0
    seen = 0
    total_ms = 0.0

    print()
    print(f"ckpt: {ckpt_path}")
    print(f"images_dir: {images_dir}")
    print("-" * 110)
    print(f"{'img':>6}  {'true':>10}  {'pred':>10}  {'conf':>6}  {'ms':>7}  top3")
    print("-" * 110)

    t_total0 = time.perf_counter()

    for i in range(1, args.n + 1):
        img_path = images_dir / f"test_{i}.jpg"
        true = TRUE_LABELS.get(i)

        if not img_path.exists():
            print(f"{i:>6}  {'(missing)':>10}  {'-':>10}  {'-':>6}  {'-':>7}  {img_path}")
            continue
        if true is None:
            print(f"{i:>6}  {'(no_gt)':>10}  {'-':>10}  {'-':>6}  {'-':>7}  {img_path}")
            continue

        img = Image.open(img_path).convert("RGB")
        x = tf(img).unsqueeze(0).to(device)

        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=1)[0].detach().cpu()

        if device.type == "cuda":
            torch.cuda.synchronize()
        dt_ms = (time.perf_counter() - t0) * 1000.0

        pred_idx = int(probs.argmax().item())
        pred = CLASSES[pred_idx]
        conf = float(probs[pred_idx].item())

        top3_idxs = torch.topk(probs, k=3).indices.tolist()
        is_top1 = pred == true
        is_top3 = any(CLASSES[j] == true for j in top3_idxs)

        seen += 1
        total_ms += dt_ms
        correct_top1 += int(is_top1)
        correct_top3 += int(is_top3)

        print(f"{i:>6}  {true:>10}  {pred:>10}  {conf:>6.3f}  {dt_ms:>7.2f}  {format_topk(probs, k=3)}")

    total_s = time.perf_counter() - t_total0
    avg_ms = (total_ms / seen) if seen else 0.0
    acc1 = (correct_top1 / seen) if seen else 0.0
    acc3 = (correct_top3 / seen) if seen else 0.0

    print("-" * 110)
    print(f"evaluated={seen}/{args.n}  top1_acc={acc1:.3f}  top3_acc={acc3:.3f}  avg_ms={avg_ms:.2f}  total_s={total_s:.2f}")


if __name__ == "__main__":
    main()
