"""V2 sentiment aggregation: confidence-weighted pooling per trading day.

Cheap proxy for the attention-pooled per-headline embeddings we float in
'future work'. Uses the existing per-headline FinBERT cache (no FinBERT
re-run needed) and adds one new feature:

    confidence_weighted_net_sentiment =
        sum_h(conf_h * (pos_h - neg_h)) / sum_h(conf_h)

Confident polarity calls dominate the daily score; near-50/50 calls fade
out. This is the same idea as attention pooling where the attention weight
is the FinBERT classifier's own confidence, except it doesn't require
caching the 768-d hidden states.

Output: results/cache/daily_sentiment_features_v2.parquet
        - same 8 columns as v1 + confidence_weighted_net_sentiment.

The v1 cache is left untouched so the original Mohamed model can be
re-evaluated against this same script.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .data_loader import CACHE_DIR, load_qqq
from .sentiment_features import PER_HEADLINE_PARQUET, aggregate_to_trading_days

DAILY_PARQUET_V2 = CACHE_DIR / "daily_sentiment_features_v2.parquet"


def _confidence_weighted_aggregate(per_headline: pd.DataFrame, trading_days: pd.Series) -> pd.DataFrame:
    """Roll headlines forward and compute the new confidence-weighted column."""
    trading_days = pd.Series(sorted(pd.to_datetime(trading_days).unique()))
    hl = per_headline.copy()
    hl["date"] = pd.to_datetime(hl["date"]).dt.normalize()
    pos_idx = trading_days.searchsorted(hl["date"].values, side="left")
    in_range = pos_idx < len(trading_days)
    hl = hl.loc[in_range].copy()
    hl["assigned_trading_day"] = trading_days.iloc[pos_idx[in_range]].values

    hl["polarity"] = hl["pos"] - hl["neg"]
    hl["w_polarity"] = hl["confidence"] * hl["polarity"]

    g = hl.groupby("assigned_trading_day").agg(
        wsum=("w_polarity", "sum"),
        csum=("confidence", "sum"),
    ).reset_index().rename(columns={"assigned_trading_day": "Date"})
    # Avoid divide-by-zero on no-news days
    g["confidence_weighted_net_sentiment"] = np.where(
        g["csum"] > 0, g["wsum"] / g["csum"], 0.0
    )
    return g[["Date", "confidence_weighted_net_sentiment"]]


def build_v2_cache(force: bool = False) -> Path:
    """Combine v1 daily features + the new confidence-weighted column."""
    if DAILY_PARQUET_V2.exists() and not force:
        return DAILY_PARQUET_V2
    if not PER_HEADLINE_PARQUET.exists():
        raise FileNotFoundError(
            f"Per-headline cache missing: {PER_HEADLINE_PARQUET}. "
            f"Run src/_run_sentiment_cache.py first."
        )

    per_hl = pd.read_parquet(PER_HEADLINE_PARQUET)
    qqq = load_qqq(drop_warmup=True)
    trading_days = qqq["Date"]

    daily_v1 = aggregate_to_trading_days(per_hl, trading_days)
    daily_extra = _confidence_weighted_aggregate(per_hl, trading_days)
    daily_v2 = daily_v1.merge(daily_extra, on="Date", how="left")
    daily_v2["confidence_weighted_net_sentiment"] = (
        daily_v2["confidence_weighted_net_sentiment"].fillna(0.0)
    )
    DAILY_PARQUET_V2.parent.mkdir(parents=True, exist_ok=True)
    daily_v2.to_parquet(DAILY_PARQUET_V2, index=False)
    return DAILY_PARQUET_V2


if __name__ == "__main__":
    path = build_v2_cache(force=True)
    print(f"wrote {path}")
    df = pd.read_parquet(path)
    print(df.head())
    print(df.describe()[["confidence_weighted_net_sentiment"]])
