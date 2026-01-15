## Training / Inference / Eval

> Note: until `src/` is packaged in `pyproject.toml`, use `PYTHONPATH=src` so imports like `from imagemodel...` work.

### Train TinyCNN (fresh)

```bash
uv run python scripts/train_cifar10_tinycnn.py --epochs 10
```

### Resume training from last checkpoint (+25 epochs)

> `--epochs` is the _total_ epoch target. Example: after 10 epochs, train to 35.

```bash
uv run python scripts/train_cifar10_tinycnn.py --resume-latest --epochs 200
```

### Inference on one image

```bash
uv run python scripts/infer_tinycnn.py --image data/test_images/test_1.jpg --ckpt models/tinycnn_cifar10/best/model.pt
```

### Inference on CIFAR-10 samples (sanity-check)

```bash
uv run python scripts/infer_tinycnn.py --use-cifar --n 10 --ckpt models/tinycnn_cifar10/best/model.pt
```

### Evaluate on all 25 test images (timing + top-3 + accuracy)

```bash
uv run python scripts/eval_tinycnn.py --ckpt models/tinycnn_cifar10/best/model.pt --images-dir data/test_images --n 25
```
