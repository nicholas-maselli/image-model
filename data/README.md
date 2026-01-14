# Data

Put datasets here. This folder is **git-ignored** to keep the repo small.

Suggested layout:

- `data/raw/`: original data as obtained (immutable)
- `data/external/`: third-party datasets
- `data/interim/`: intermediate transforms (optional)
- `data/processed/`: final training-ready data

Tips:

- Store *how to download/prepare* data (links + scripts) in `scripts/`, not the data itself.
- If you want versioned data, consider DVC later.
