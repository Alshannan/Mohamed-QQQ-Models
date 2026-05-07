"""V2 dataset assembly: adds confidence_weighted_net_sentiment as feature #15.

Wraps the v1 loader but reads from daily_sentiment_features_v2.parquet so the
v1 cache and the v1 model's pipeline stay byte-identical.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from .data_loader import (
    HORIZONS,
    PRICE_FEATURES,
    SENTIMENT_FEATURES as SENTIMENT_FEATURES_V1,
    add_targets,
    build_windows,
    chronological_split,
    load_qqq,
    FittedScalers,
)
from .sentiment_features_v2 import DAILY_PARQUET_V2, build_v2_cache

EXTRA_FEATURE = "confidence_weighted_net_sentiment"
SENTIMENT_FEATURES_V2 = SENTIMENT_FEATURES_V1 + [EXTRA_FEATURE]
ALL_FEATURES_V2 = PRICE_FEATURES + SENTIMENT_FEATURES_V2  # 15 columns


def fit_scalers_v2(df: pd.DataFrame, train_idx: np.ndarray) -> FittedScalers:
    """Same scalers as v1 (the new feature is already in [-1, 1])."""
    ohlcv = MinMaxScaler()
    ohlcv.fit(df.iloc[train_idx][["Open", "High", "Low", "Close", "Volume"]].values)
    macd = StandardScaler()
    macd.fit(df.iloc[train_idx][["MACD"]].values)
    hc = StandardScaler()
    hc.fit(np.log1p(df.iloc[train_idx][["headline_count"]].values))
    return FittedScalers(ohlcv=ohlcv, macd=macd, headline_count=hc)


def transform_features_v2(df: pd.DataFrame, scalers: FittedScalers) -> np.ndarray:
    out = pd.DataFrame(index=df.index)
    ohlcv = scalers.ohlcv.transform(df[["Open", "High", "Low", "Close", "Volume"]].values)
    out["Open"], out["High"], out["Low"], out["Close"], out["Volume"] = ohlcv.T
    out["MACD"] = scalers.macd.transform(df[["MACD"]].values).ravel()

    for col in ["finbert_positive", "finbert_negative", "finbert_neutral", "finbert_confidence"]:
        out[col] = df[col].values
    out["headline_count"] = scalers.headline_count.transform(
        np.log1p(df[["headline_count"]].values)
    ).ravel()
    out["no_news_flag"] = df["no_news_flag"].values
    out["net_sentiment_ma3"] = df["net_sentiment_ma3"].values
    out["net_sentiment_ma7"] = df["net_sentiment_ma7"].values
    out[EXTRA_FEATURE] = df[EXTRA_FEATURE].values

    return out[ALL_FEATURES_V2].values.astype(np.float32)


def assemble_dataset_v2(
    window: int = 20,
    horizons: Iterable[int] = HORIZONS,
    sentiment_path: Path | None = None,
):
    """Drop-in v2 assembly returning 15-feature windows."""
    if sentiment_path is None:
        sentiment_path = build_v2_cache(force=False)

    price = load_qqq(drop_warmup=True)
    price = add_targets(price, horizons=horizons)
    sent = pd.read_parquet(sentiment_path)
    sent["Date"] = pd.to_datetime(sent["Date"])
    df = price.merge(sent, on="Date", how="left")

    for col in ["finbert_positive", "finbert_negative", "finbert_neutral", "finbert_confidence",
                EXTRA_FEATURE, "net_sentiment_ma3", "net_sentiment_ma7"]:
        df[col] = df[col].fillna(0.0)
    df["headline_count"] = df["headline_count"].fillna(0).astype(int)
    df["no_news_flag"] = (df["headline_count"] == 0).astype(int)

    split = chronological_split(df["Date"])
    scalers = fit_scalers_v2(df, split.train)
    features = transform_features_v2(df, scalers)
    target_cols = [f"target_t{k}" for k in horizons]
    targets = df[target_cols].values.astype(np.float32)

    Xtr, ytr, idxtr = build_windows(features, targets, window, split.train)
    Xva, yva, idxva = build_windows(features, targets, window, split.val)
    Xte, yte, idxte = build_windows(features, targets, window, split.test)

    return {
        "X_train": Xtr, "y_train": ytr, "idx_train": idxtr,
        "X_val": Xva,   "y_val": yva,   "idx_val": idxva,
        "X_test": Xte,  "y_test": yte,  "idx_test": idxte,
        "scalers": scalers,
        "split": split,
        "df": df,
        "feature_names": ALL_FEATURES_V2,
        "target_cols": target_cols,
    }
