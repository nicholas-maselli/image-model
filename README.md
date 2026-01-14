# image-model

Vision model repo organized for learning + iteration.

## Recommended structure

- `src/imagemodel/`: reusable library code (datasets, models, training loops, utils)
- `configs/`: experiment configs (YAML/JSON/TOML) for reproducible runs
- `scripts/`: CLI entrypoints (train/eval/export/sweep)
- `notebooks/`: exploration/learning; promote stable work into `src/` + `scripts/`
- `data/`: datasets (ignored by git; keep docs in `data/README.md`)
- `models/`: checkpoints/weights (ignored by git; keep docs in `models/README.md`)
- `runs/`: logs/metrics/artifacts (ignored by git; keep docs in `runs/README.md`)
- `reports/`: results you *do* want in git (plots, tables, writeups)
- `docs/`: longer-form notes/design docs
- `tests/`: unit/integration tests for `src/`

## Workflow that keeps things clean

1. Explore quickly in `notebooks/`
2. When something works, move it into `src/imagemodel/` (reusable) + `scripts/` (runnable)
3. Keep datasets/checkpoints/logs out of git (`data/`, `models/`, `runs/` are ignored)
4. Save shareable results in `reports/` (figures, short summaries)
