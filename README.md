# image-model

Vision model repo organized for learning + iteration.

## Recommended structure

-   `src/`: reusable library code (datasets, models, training loops, utils)
-   `scripts/`: CLI entrypoints (train/eval/export/sweep)
-   `data/`: datasets (ignored by git; keep docs in `data/README.md`)
-   `models/`: checkpoints/weights (ignored by git; keep docs in `models/README.md`)
-   `tests/`: unit/integration tests for `src/`

## Running the Code

# Training

uv run python src/scripts/train/train.py \
 --dataset cifar10 \
 --model micro \
 --steps 10000

# Training (No GPU Pre Evaluation)

CUDA_VISIBLE_DEVICES=0 uv run python src/scripts/train/train.py --dataset cifar10 --model test_candidate_cuda0 --steps 10000

CUDA_VISIBLE_DEVICES=1 uv run python src/scripts/train/train.py --dataset cifar10 --model test_candidate_cuda1 --steps 10000

## Evals:

# KiloCNN
CUDA_VISIBLE_DEVICES=0 uv run python src/scripts/train/train.py --dataset cifar10 --model kilo --steps 10000
final eval step 10000  test_loss=0.4458 test_acc=0.8804  best_acc=0.8859  time=0.5s

CUDA_VISIBLE_DEVICES=0 uv run python src/scripts/train/train.py --dataset cifar10 --model kilo --steps 50000
final eval step 50000  test_loss=0.4654 test_acc=0.9212  best_acc=0.9212  time=0.5s

# StandardCNN
CUDA_VISIBLE_DEVICES=0 uv run python src/scripts/train/train.py --dataset cifar10 --model standard --steps 10000
final eval step 10000  test_loss=0.4856 test_acc=0.8571  best_acc=0.8571  time=0.4s

CUDA_VISIBLE_DEVICES=0 uv run python src/scripts/train/train.py --dataset cifar10 --model standard --steps 50000
final eval step 50000  test_loss=0.5824 test_acc=0.8895  best_acc=0.8982  time=0.5s

# TestCandidate1Res
CUDA_VISIBLE_DEVICES=0 uv run python src/scripts/train/train.py --dataset cifar10 --model test_candidate_cuda0 --steps 10000
final eval step 10000  test_loss=0.4210 test_acc=0.8729  best_acc=0.8729  time=0.5s

CUDA_VISIBLE_DEVICES=0 uv run python src/scripts/train/train.py --dataset cifar10 --model test_candidate_cuda0 --steps 50000
final eval step 50000  test_loss=0.5488 test_acc=0.8991  best_acc=0.8991  time=0.5s

# TestCandidate2GroupNorm
CUDA_VISIBLE_DEVICES=1 uv run python src/scripts/train/train.py --dataset cifar10 --model test_candidate_cuda1 --steps 10000
final eval step 10000  test_loss=0.4816 test_acc=0.8365  best_acc=0.8365  time=0.4s

CUDA_VISIBLE_DEVICES=1 uv run python src/scripts/train/train.py --dataset cifar10 --model test_candidate_cuda1 --steps 50000
final eval step 50000  test_loss=0.5043 test_acc=0.8815  best_acc=0.8885  time=0.5s

# TestCandidateBasicBlockRestnet
CUDA_VISIBLE_DEVICES=0 uv run python src/scripts/train/train.py --dataset cifar10 --model test_candidate_cuda0 --steps 50000
final eval step 50000  test_loss=0.5488 test_acc=0.8991  best_acc=0.8991  time=0.5s

# TestCandidateBigBasicBlockRestnet
CUDA_VISIBLE_DEVICES=0 uv run python src/scripts/train/train.py --dataset cifar10 --model test_candidate_cuda0 --steps 50000
final eval step 50000  test_loss=0.4776 test_acc=0.9236  best_acc=0.9276  time=0.7s

# TestCandidatePreActivation
CUDA_VISIBLE_DEVICES=0 uv run python src/scripts/train/train.py --dataset cifar10 --model test_candidate_cuda0 --steps 50000
eval step 50000  test_loss=0.5593 test_acc=0.9166  best_acc=0.9290  time=0.7s
model=test_candidate_cuda0 final eval step 50000  test_loss=0.5593 test_acc=0.9166  best_acc=0.9290  time=0.7s
