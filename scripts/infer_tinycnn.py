import argparse
from pathlib import Path

import torch
from PIL import Image
from torchvision import datasets, transforms

from imagemodel.models.tiny_cnn import TinyCNN

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)

CLASSES = ("airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck")

def load_model(ckpt_path: str, device: torch.device) -> TinyCNN:
    model = TinyCNN(num_classes=10).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, default="models/tinycnn_cifar10/best/model.pt")
    p.add_argument("--image", type=str, default=None, help="Path to an image file to classify")
    p.add_argument("--use-cifar", action="store_true", help="Run on CIFAR-10 test samples instead of a file")
    p.add_argument("--n", type=int, default=5, help="How many CIFAR test samples to run")
    p.add_argument("--data-root", type=str, default="data/raw")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.ckpt, device)

    tf = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    if args.use_cifar:
        ds = datasets.CIFAR10(root=args.data_root, train=False, download=False, transform=tf)
        for i in range(args.n):
            x, y = ds[i]
            x = x.unsqueeze(0).to(device)
            with torch.no_grad():
                logits = model(x)
                probs = torch.softmax(logits, dim=1)[0]
                pred = int(probs.argmax().item())
            print(f"idx={i}  true={CLASSES[y]}  pred={CLASSES[pred]}  conf={probs[pred].item():.3f}")
        return

    if args.image is None:
        raise SystemExit("Provide --image path/to/file.jpg OR use --use-cifar")

    img = Image.open(args.image).convert("RGB")
    x = tf(img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0]
        pred = int(probs.argmax().item())

    print(f"pred={CLASSES[pred]}  conf={probs[pred].item():.3f}")
    # Optional: print top-3
    topk = torch.topk(probs, k=3)
    for score, cls_idx in zip(topk.values.tolist(), topk.indices.tolist()):
        print(f"  {CLASSES[cls_idx]}: {score:.3f}")

if __name__ == "__main__":
    main()
