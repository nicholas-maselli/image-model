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

CUDA_VISIBLE_DEVICES=0 uv run python src/scripts/train/train.py --dataset cifar10 --model test_candidate_0 --steps 10000

CUDA_VISIBLE_DEVICES=1 uv run python src/scripts/train/train.py --dataset cifar10 --model test_candidate_1 --steps 10000

## Evals:

# StandardCNN
final eval step 10000  test_loss=0.4856 test_acc=0.8571  best_acc=0.8571  time=0.4s

#