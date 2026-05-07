"""One-shot script: generate FinBERT cache. Invoked from a notebook or shell."""
from __future__ import annotations

import sys
from pathlib import Path

# Allow running as `python src/_run_sentiment_cache.py`
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.sentiment_features import ensure_sentiment_cache


def main():
    per, daily, used = ensure_sentiment_cache(force=False)
    print(f"per-headline cache: {per}")
    print(f"daily cache:        {daily}")
    print(f"model used:         {used}")


if __name__ == "__main__":
    main()
