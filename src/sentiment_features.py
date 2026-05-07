"""FinBERT inference + daily aggregation for the headline corpus.

Cache the per-headline scores (so we never re-run FinBERT) and build the
8-dim daily sentiment feature block keyed on trading day. Weekend / holiday
headlines roll forward to the next trading day.

If the FinBERT download fails, the brief authorizes falling back to a
distilled sentiment model. We try ProsusAI/finbert first, then
mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from .data_loader import (
    CACHE_DIR,
    HEADLINES_PATH,
    load_headlines,
    load_qqq,
)

PER_HEADLINE_PARQUET = CACHE_DIR / "headline_finbert_scores.parquet"
DAILY_PARQUET = CACHE_DIR / "daily_sentiment_features.parquet"

FINBERT_MODELS = (
    "ProsusAI/finbert",
    "mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis",
)


def _label_to_index(label: str, model_name: str) -> int:
    """Map model-specific label strings to (positive, negative, neutral) idx."""
    label = label.lower()
    if label in {"positive", "label_2"} or label.startswith("pos"):
        return 0
    if label in {"negative", "label_0"} or label.startswith("neg"):
        return 1
    return 2  # neutral / label_1 / unknown


def score_headlines(
    headlines: pd.DataFrame,
    batch_size: int = 32,
    max_length: int = 96,
    progress_every: int = 50,
) -> tuple[pd.DataFrame, str]:
    """Run a financial-sentiment classifier over all headlines.

    Returns (per-headline scores df, model_name_used).
    The output df has columns: date, headline, pos, neg, neu, confidence.
    """
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    last_err: Exception | None = None
    model = tokenizer = used = None
    for name in FINBERT_MODELS:
        try:
            tokenizer = AutoTokenizer.from_pretrained(name)
            model = AutoModelForSequenceClassification.from_pretrained(name)
            used = name
            break
        except Exception as e:  # network or compatibility failure
            last_err = e
            print(f"[sentiment] Could not load {name}: {e}")
    if model is None or tokenizer is None:
        raise RuntimeError(f"All sentiment models failed; last error: {last_err}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()
    print(f"[sentiment] Using {used} on {device}")

    # Build a label permutation that maps model output indices to (pos,neg,neu).
    id2label = {int(i): str(l) for i, l in model.config.id2label.items()}
    perm = [None, None, None]
    for idx, lbl in id2label.items():
        slot = _label_to_index(lbl, used)
        perm[slot] = idx
    if any(p is None for p in perm):
        # Fallback to natural order if labels are unrecognized.
        perm = [0, 1, 2]
    perm = np.asarray(perm)

    texts = headlines["headline"].astype(str).tolist()
    n = len(texts)
    all_probs = np.zeros((n, 3), dtype=np.float32)

    start = time.time()
    with torch.no_grad():
        for b in range(0, n, batch_size):
            chunk = texts[b : b + batch_size]
            enc = tokenizer(
                chunk,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(device)
            logits = model(**enc).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            # Reorder columns -> [pos, neg, neu]
            all_probs[b : b + len(chunk)] = probs[:, perm]
            done = b // batch_size
            if done and done % progress_every == 0:
                elapsed = time.time() - start
                rate = (b + len(chunk)) / max(elapsed, 1e-6)
                eta = (n - b - len(chunk)) / max(rate, 1e-6)
                print(
                    f"[sentiment] {b + len(chunk)}/{n} "
                    f"({rate:.1f} hl/s, ETA {eta/60:.1f}min)"
                )

    out = headlines[["date", "headline"]].copy()
    out["pos"] = all_probs[:, 0]
    out["neg"] = all_probs[:, 1]
    out["neu"] = all_probs[:, 2]
    out["confidence"] = all_probs.max(axis=1)
    return out, used


def aggregate_to_trading_days(
    per_headline: pd.DataFrame,
    trading_days: pd.Series,
) -> pd.DataFrame:
    """Roll headlines forward to the next trading day, then mean-aggregate.

    `trading_days` is a chronologically sorted Series of pd.Timestamp values
    (the QQQ Date column after warmup drop).
    """
    trading_days = pd.Series(sorted(pd.to_datetime(trading_days).unique()))
    # For each headline date d, find the smallest trading day >= d.
    hl = per_headline.copy()
    hl["date"] = pd.to_datetime(hl["date"]).dt.normalize()
    pos_idx = trading_days.searchsorted(hl["date"].values, side="left")
    in_range = pos_idx < len(trading_days)
    hl = hl.loc[in_range].copy()
    hl["assigned_trading_day"] = trading_days.iloc[pos_idx[in_range]].values

    grouped = hl.groupby("assigned_trading_day").agg(
        finbert_positive=("pos", "mean"),
        finbert_negative=("neg", "mean"),
        finbert_neutral=("neu", "mean"),
        finbert_confidence=("confidence", "mean"),
        headline_count=("headline", "count"),
    ).reset_index().rename(columns={"assigned_trading_day": "Date"})

    full = pd.DataFrame({"Date": trading_days})
    daily = full.merge(grouped, on="Date", how="left")
    fill_zero = ["finbert_positive", "finbert_negative", "finbert_neutral",
                 "finbert_confidence", "headline_count"]
    for c in fill_zero:
        daily[c] = daily[c].fillna(0.0)
    daily["headline_count"] = daily["headline_count"].astype(int)
    daily["no_news_flag"] = (daily["headline_count"] == 0).astype(int)
    daily["net_sentiment"] = daily["finbert_positive"] - daily["finbert_negative"]
    daily["net_sentiment_ma3"] = (
        daily["net_sentiment"].rolling(3, min_periods=1).mean()
    )
    daily["net_sentiment_ma7"] = (
        daily["net_sentiment"].rolling(7, min_periods=1).mean()
    )
    return daily


def ensure_sentiment_cache(force: bool = False) -> tuple[Path, Path, str | None]:
    """Build the per-headline + daily caches if missing. Returns paths."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    used_model: str | None = None
    if force or not PER_HEADLINE_PARQUET.exists():
        headlines = load_headlines()
        scored, used_model = score_headlines(headlines)
        scored.to_parquet(PER_HEADLINE_PARQUET, index=False)
    else:
        scored = pd.read_parquet(PER_HEADLINE_PARQUET)

    if force or not DAILY_PARQUET.exists():
        qqq = load_qqq(drop_warmup=True)
        daily = aggregate_to_trading_days(scored, qqq["Date"])
        daily.to_parquet(DAILY_PARQUET, index=False)

    return PER_HEADLINE_PARQUET, DAILY_PARQUET, used_model
