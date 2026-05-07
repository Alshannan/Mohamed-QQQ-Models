"""V3 per-trading-day headline sequences for MANA-Net attention pooling.

For each trading day after roll-forward, build a fixed-size padded tensor
of shape (MAX_HEADLINES, 4) holding [pos, neg, neu, confidence] for the
day's headlines, plus a boolean mask. Cap at MAX_HEADLINES=16 (~99th
percentile of headlines-per-trading-day after weekend/holiday roll-
forward); days with more headlines keep the most recent 16 (heuristic:
sort by per-headline original date, drop oldest).

Output: results/cache/per_day_headlines_v3.npz with arrays:
  headlines  (n_days, 16, 4) float32
  mask       (n_days, 16)    bool   (True where padding)
  dates      (n_days,)       datetime64[ns]
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .data_loader import CACHE_DIR, load_qqq
from .sentiment_features import PER_HEADLINE_PARQUET

MAX_HEADLINES = 16
SENTIMENT_DIM = 4  # pos, neg, neu, confidence
HEADLINE_NPZ = CACHE_DIR / "per_day_headlines_v3.npz"


def build_headline_sequences(force: bool = False) -> Path:
    if HEADLINE_NPZ.exists() and not force:
        return HEADLINE_NPZ
    if not PER_HEADLINE_PARQUET.exists():
        raise FileNotFoundError(
            f"Missing {PER_HEADLINE_PARQUET}. Run src/_run_sentiment_cache.py first."
        )

    per_hl = pd.read_parquet(PER_HEADLINE_PARQUET)
    qqq = load_qqq(drop_warmup=True)
    trading_days = pd.Series(sorted(pd.to_datetime(qqq["Date"]).dt.normalize().unique()))

    hl = per_hl.copy()
    hl["date"] = pd.to_datetime(hl["date"]).dt.normalize()
    pos_idx = trading_days.searchsorted(hl["date"].values, side="left")
    in_range = pos_idx < len(trading_days)
    hl = hl.loc[in_range].copy()
    hl["assigned"] = trading_days.iloc[pos_idx[in_range]].values

    # Sort within each assigned trading day so we can keep the most-recent N
    hl = hl.sort_values(["assigned", "date"]).reset_index(drop=True)

    n_days = len(trading_days)
    headlines = np.zeros((n_days, MAX_HEADLINES, SENTIMENT_DIM), dtype=np.float32)
    mask = np.ones((n_days, MAX_HEADLINES), dtype=bool)  # True = padding

    day_to_idx = {pd.Timestamp(d): i for i, d in enumerate(trading_days)}
    for assigned, group in hl.groupby("assigned", sort=False):
        idx = day_to_idx[pd.Timestamp(assigned)]
        # Keep the most recent MAX_HEADLINES (already sorted ascending by date)
        chosen = group.tail(MAX_HEADLINES)
        n = len(chosen)
        headlines[idx, :n, 0] = chosen["pos"].values
        headlines[idx, :n, 1] = chosen["neg"].values
        headlines[idx, :n, 2] = chosen["neu"].values
        headlines[idx, :n, 3] = chosen["confidence"].values
        mask[idx, :n] = False  # False = real data

    HEADLINE_NPZ.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        HEADLINE_NPZ,
        headlines=headlines,
        mask=mask,
        dates=np.array(trading_days, dtype="datetime64[ns]"),
    )
    return HEADLINE_NPZ


if __name__ == "__main__":
    p = build_headline_sequences(force=True)
    z = np.load(p, allow_pickle=False)
    print(f"wrote {p}")
    print(f"  headlines: {z['headlines'].shape}, mask: {z['mask'].shape}, dates: {z['dates'].shape}")
    print(f"  fraction of slots that are real data: {(~z['mask']).mean():.3f}")
    print(f"  mean headlines per day: {(~z['mask']).sum(axis=1).mean():.2f}")
    print(f"  max headlines per day used: {(~z['mask']).sum(axis=1).max()}")
