"""V3 dataset assembly.

Adds three blocks on top of v2:
  - 9 technical features (RSI, Bollinger %B, ATR/Close, OBV-z, vol(20),
    lag returns, day-of-week sin/cos)
  - 5 macro features (VIX close + log-return, ^TNX yield + diff, DXY log-return)
  - per-trading-day padded headline tensor (MAX_HEADLINES, 4) for MANA-Net

Returns BOTH the standard windowed feature matrix X_features (B, T, F)
and the per-window headline tensor X_headlines (B, T, MAX_HEADLINES, 4)
plus its mask (B, T, MAX_HEADLINES). v3 keeps the daily-aggregated
sentiment features so MANA-Net is strictly additive (you can ablate by
zeroing the attention output).
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
    add_targets,
    build_windows,
    chronological_split,
    load_qqq,
    FittedScalers,
)
from .data_loader_v2 import EXTRA_FEATURE as V2_EXTRA_FEATURE
from .headline_sequences import HEADLINE_NPZ, MAX_HEADLINES, SENTIMENT_DIM, build_headline_sequences
from .macro_features import (
    ALL_V3_EXTRA as MACRO_AND_TECH,
    MACRO_PARQUET,
    build_v3_macro_cache,
)
from .sentiment_features_v2 import DAILY_PARQUET_V2, build_v2_cache

V2_SENT_COLS = [
    "finbert_positive", "finbert_negative", "finbert_neutral", "finbert_confidence",
    "headline_count", "no_news_flag", "net_sentiment_ma3", "net_sentiment_ma7",
    V2_EXTRA_FEATURE,  # confidence_weighted_net_sentiment
]
ALL_FEATURES_V3 = PRICE_FEATURES + V2_SENT_COLS + MACRO_AND_TECH


def fit_scalers_v3(df: pd.DataFrame, train_idx: np.ndarray) -> FittedScalers:
    ohlcv = MinMaxScaler()
    ohlcv.fit(df.iloc[train_idx][["Open", "High", "Low", "Close", "Volume"]].values)
    macd = StandardScaler()
    macd.fit(df.iloc[train_idx][["MACD"]].values)
    hc = StandardScaler()
    hc.fit(np.log1p(df.iloc[train_idx][["headline_count"]].values))
    return FittedScalers(ohlcv=ohlcv, macd=macd, headline_count=hc)


def _v3_macro_zscore_params(df: pd.DataFrame, train_idx: np.ndarray) -> dict:
    """Per-column mean/std fit on TRAIN ONLY, used for vix_close and tnx_yield."""
    train_slice = df.iloc[train_idx]
    out = {}
    for col in ("vix_close", "tnx_yield"):
        out[col] = (float(train_slice[col].mean()), float(train_slice[col].std() + 1e-6))
    return out


def transform_features_v3(df: pd.DataFrame, scalers: FittedScalers, train_idx: np.ndarray) -> np.ndarray:
    out = pd.DataFrame(index=df.index)
    ohlcv = scalers.ohlcv.transform(df[["Open", "High", "Low", "Close", "Volume"]].values)
    out["Open"], out["High"], out["Low"], out["Close"], out["Volume"] = ohlcv.T
    out["MACD"] = scalers.macd.transform(df[["MACD"]].values).ravel()

    # v2 sentiment block (already in [0,1] or roughly [-1,1])
    for col in ["finbert_positive", "finbert_negative", "finbert_neutral",
                "finbert_confidence", "no_news_flag", "net_sentiment_ma3",
                "net_sentiment_ma7", V2_EXTRA_FEATURE]:
        out[col] = df[col].values
    out["headline_count"] = scalers.headline_count.transform(
        np.log1p(df[["headline_count"]].values)
    ).ravel()

    # v3 macro/technical block (most are already in reasonable ranges).
    # vix_close and tnx_yield need z-scoring; do it with TRAIN-ONLY stats.
    zparams = _v3_macro_zscore_params(df, train_idx)
    for col in MACRO_AND_TECH:
        if col in zparams:
            mean, std = zparams[col]
            out[col] = ((df[col].values - mean) / std).clip(-5, 5)
        else:
            out[col] = df[col].values

    return out[ALL_FEATURES_V3].values.astype(np.float32)


def _build_headline_windows(
    headlines: np.ndarray, mask: np.ndarray, window: int, end_idx: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Slice (n_days, MAX, 4) into windowed form (B, T, MAX, 4) using end indices.

    headlines, mask are aligned with the underlying df ordering.
    end_idx is the per-window end index (`t`); window covers [t-T+1, t].
    """
    Xh = np.empty((len(end_idx), window, MAX_HEADLINES, SENTIMENT_DIM), dtype=np.float32)
    Xm = np.empty((len(end_idx), window, MAX_HEADLINES), dtype=bool)
    for i, t in enumerate(end_idx):
        Xh[i] = headlines[t - window + 1 : t + 1]
        Xm[i] = mask[t - window + 1 : t + 1]
    return Xh, Xm


def assemble_dataset_v3(
    window: int = 20,
    horizons: Iterable[int] = HORIZONS,
):
    """Build the v3 (multi-tensor) dataset.

    Returns a dict with X_train, X_val, X_test (base features), plus
    H_train/H_val/H_test (headlines) and M_train/M_val/M_test (masks).
    """
    sentiment_path = build_v2_cache(force=False)
    macro_path = build_v3_macro_cache(force=False)
    headline_npz = build_headline_sequences(force=False)

    price = load_qqq(drop_warmup=True)
    price = add_targets(price, horizons=horizons)

    sent = pd.read_parquet(sentiment_path); sent["Date"] = pd.to_datetime(sent["Date"])
    macro = pd.read_parquet(macro_path);    macro["Date"] = pd.to_datetime(macro["Date"])

    df = price.merge(sent, on="Date", how="left").merge(macro, on="Date", how="left")
    for col in ["finbert_positive", "finbert_negative", "finbert_neutral",
                "finbert_confidence", "net_sentiment_ma3", "net_sentiment_ma7",
                V2_EXTRA_FEATURE]:
        df[col] = df[col].fillna(0.0)
    df["headline_count"] = df["headline_count"].fillna(0).astype(int)
    df["no_news_flag"] = (df["headline_count"] == 0).astype(int)
    for col in MACRO_AND_TECH:
        df[col] = df[col].fillna(0.0)

    # Headline tensors keyed by Date - must match df ordering exactly
    npz = np.load(headline_npz, allow_pickle=False)
    headlines, mask, hl_dates = npz["headlines"], npz["mask"], npz["dates"]
    hl_date_to_idx = {pd.Timestamp(d): i for i, d in enumerate(hl_dates)}
    df_dates = pd.to_datetime(df["Date"]).dt.normalize().values
    head_idx = np.array([hl_date_to_idx[pd.Timestamp(d)] for d in df_dates], dtype=np.int64)
    headlines_aligned = headlines[head_idx]
    mask_aligned = mask[head_idx]

    split = chronological_split(df["Date"])
    scalers = fit_scalers_v3(df, split.train)
    features = transform_features_v3(df, scalers, split.train)
    target_cols = [f"target_t{k}" for k in horizons]
    targets = df[target_cols].values.astype(np.float32)

    Xtr, ytr, idxtr = build_windows(features, targets, window, split.train)
    Xva, yva, idxva = build_windows(features, targets, window, split.val)
    Xte, yte, idxte = build_windows(features, targets, window, split.test)

    Htr, Mtr = _build_headline_windows(headlines_aligned, mask_aligned, window, idxtr)
    Hva, Mva = _build_headline_windows(headlines_aligned, mask_aligned, window, idxva)
    Hte, Mte = _build_headline_windows(headlines_aligned, mask_aligned, window, idxte)

    return {
        "X_train": Xtr, "y_train": ytr, "idx_train": idxtr, "H_train": Htr, "M_train": Mtr,
        "X_val":   Xva, "y_val":   yva, "idx_val":   idxva, "H_val":   Hva, "M_val":   Mva,
        "X_test":  Xte, "y_test":  yte, "idx_test":  idxte, "H_test":  Hte, "M_test":  Mte,
        "scalers": scalers,
        "split": split,
        "df": df,
        "feature_names": ALL_FEATURES_V3,
        "target_cols": target_cols,
        "max_headlines": MAX_HEADLINES,
        "sentiment_dim": SENTIMENT_DIM,
    }


if __name__ == "__main__":
    data = assemble_dataset_v3(window=20)
    print(f"X_train: {data['X_train'].shape}  H_train: {data['H_train'].shape}  M_train: {data['M_train'].shape}")
    print(f"X_test:  {data['X_test'].shape}   H_test:  {data['H_test'].shape}   M_test:  {data['M_test'].shape}")
    print(f"feature count: {len(data['feature_names'])}")
    print(f"feature_names: {data['feature_names']}")
