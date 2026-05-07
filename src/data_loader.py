"""Shared data loading, splitting, scaling, and windowing utilities.

Same loader is used by the baseline (6 price features) and Mohamed's model
(14 features), so the train/val/test split and target construction are
guaranteed identical across both runs.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
CACHE_DIR = PROJECT_ROOT / "results" / "cache"
MODELS_DIR = PROJECT_ROOT / "results" / "models"
PLOTS_DIR = PROJECT_ROOT / "results" / "plots"

QQQ_PATH = DATA_DIR / "QQQ_2000_2024_with_MACD.csv"
HEADLINES_PATH = DATA_DIR / "financial_headlines_2000_2024.csv"

WARMUP_DROP_ROWS = 30  # MACD 12/26/9 warmup
HORIZONS = (1, 3, 5)
SEED = 42

PRICE_FEATURES = ["Open", "High", "Low", "Close", "Volume", "MACD"]
SENTIMENT_FEATURES = [
    "finbert_positive",
    "finbert_negative",
    "finbert_neutral",
    "finbert_confidence",
    "headline_count",
    "no_news_flag",
    "net_sentiment_ma3",
    "net_sentiment_ma7",
]
ALL_FEATURES = PRICE_FEATURES + SENTIMENT_FEATURES

TRAIN_END = pd.Timestamp("2017-06-30")
VAL_END = pd.Timestamp("2021-03-31")
TEST_END = pd.Timestamp("2024-12-31")


def load_qqq(drop_warmup: bool = True) -> pd.DataFrame:
    """Load the QQQ price file. Optionally drop MACD warmup rows."""
    df = pd.read_csv(QQQ_PATH)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    if drop_warmup:
        df = df.iloc[WARMUP_DROP_ROWS:].reset_index(drop=True)
    return df


def load_headlines() -> pd.DataFrame:
    """Load the headline file. Date is parsed; rows with bad dates dropped."""
    df = pd.read_csv(HEADLINES_PATH)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "headline"]).reset_index(drop=True)
    return df


def add_targets(df: pd.DataFrame, horizons: Iterable[int] = HORIZONS) -> pd.DataFrame:
    """Add forward-return targets r_{t+k} = (Close[t+k] - Close[t]) / Close[t]."""
    out = df.copy()
    for k in horizons:
        out[f"target_t{k}"] = out["Close"].shift(-k) / out["Close"] - 1.0
    return out


@dataclass
class SplitIndex:
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray
    train_dates: tuple[pd.Timestamp, pd.Timestamp]
    val_dates: tuple[pd.Timestamp, pd.Timestamp]
    test_dates: tuple[pd.Timestamp, pd.Timestamp]


def chronological_split(dates: pd.Series) -> SplitIndex:
    """Return integer index arrays for train/val/test by calendar bounds."""
    train_mask = dates <= TRAIN_END
    val_mask = (dates > TRAIN_END) & (dates <= VAL_END)
    test_mask = (dates > VAL_END) & (dates <= TEST_END)
    train_idx = np.where(train_mask)[0]
    val_idx = np.where(val_mask)[0]
    test_idx = np.where(test_mask)[0]
    return SplitIndex(
        train=train_idx,
        val=val_idx,
        test=test_idx,
        train_dates=(dates.iloc[train_idx[0]], dates.iloc[train_idx[-1]]),
        val_dates=(dates.iloc[val_idx[0]], dates.iloc[val_idx[-1]]),
        test_dates=(dates.iloc[test_idx[0]], dates.iloc[test_idx[-1]]),
    )


@dataclass
class FittedScalers:
    ohlcv: MinMaxScaler
    macd: StandardScaler
    headline_count: StandardScaler | None  # None for the baseline


def fit_scalers(
    df: pd.DataFrame,
    train_idx: np.ndarray,
    use_sentiment: bool,
) -> FittedScalers:
    """Fit scalers on the train slice ONLY."""
    ohlcv = MinMaxScaler()
    ohlcv.fit(df.iloc[train_idx][["Open", "High", "Low", "Close", "Volume"]].values)
    macd = StandardScaler()
    macd.fit(df.iloc[train_idx][["MACD"]].values)

    hc = None
    if use_sentiment:
        hc = StandardScaler()
        hc.fit(np.log1p(df.iloc[train_idx][["headline_count"]].values))
    return FittedScalers(ohlcv=ohlcv, macd=macd, headline_count=hc)


def transform_features(
    df: pd.DataFrame, scalers: FittedScalers, use_sentiment: bool
) -> np.ndarray:
    """Apply fitted scalers and return the per-row feature matrix.

    Order matches PRICE_FEATURES (+ SENTIMENT_FEATURES if use_sentiment).
    """
    out = pd.DataFrame(index=df.index)
    ohlcv = scalers.ohlcv.transform(df[["Open", "High", "Low", "Close", "Volume"]].values)
    out["Open"], out["High"], out["Low"], out["Close"], out["Volume"] = ohlcv.T
    out["MACD"] = scalers.macd.transform(df[["MACD"]].values).ravel()

    if use_sentiment:
        for col in ["finbert_positive", "finbert_negative", "finbert_neutral", "finbert_confidence"]:
            out[col] = df[col].values  # already in [0,1]
        if scalers.headline_count is None:
            raise RuntimeError("headline_count scaler missing")
        out["headline_count"] = scalers.headline_count.transform(
            np.log1p(df[["headline_count"]].values)
        ).ravel()
        out["no_news_flag"] = df["no_news_flag"].values
        out["net_sentiment_ma3"] = df["net_sentiment_ma3"].values
        out["net_sentiment_ma7"] = df["net_sentiment_ma7"].values

    cols = ALL_FEATURES if use_sentiment else PRICE_FEATURES
    return out[cols].values.astype(np.float32)


def build_windows(
    features: np.ndarray,
    targets: np.ndarray,
    window: int,
    valid_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Slide a window over `features` and pull aligned `targets`.

    A window ending at index `t` (inclusive) requires:
      - t - window + 1 >= 0  (enough history)
      - all 3 horizon targets at row `t` are finite (no shift-induced NaN)
      - `t` falls inside `valid_indices` (the split mask)
    Returns (X, y, end_idx_kept) where end_idx_kept aligns each row of X/y to
    the original DataFrame index - useful for downstream date plots.
    """
    valid_set = set(valid_indices.tolist())
    keep_X = []
    keep_y = []
    keep_idx = []
    finite = np.all(np.isfinite(targets), axis=1)
    for t in range(window - 1, features.shape[0]):
        if t not in valid_set:
            continue
        if not finite[t]:
            continue
        keep_X.append(features[t - window + 1 : t + 1])
        keep_y.append(targets[t])
        keep_idx.append(t)
    return (
        np.stack(keep_X).astype(np.float32),
        np.stack(keep_y).astype(np.float32),
        np.array(keep_idx, dtype=np.int64),
    )


def assemble_dataset(
    use_sentiment: bool,
    window: int,
    horizons: Iterable[int] = HORIZONS,
    sentiment_path: Path | None = None,
):
    """Top-level convenience: load price (+ sentiment), split, scale, window.

    Returns a dict with X/y/end_idx for train/val/test plus the scaler bundle
    and the underlying merged DataFrame (for date lookups in evaluation).
    """
    price = load_qqq(drop_warmup=True)
    price = add_targets(price, horizons=horizons)

    if use_sentiment:
        if sentiment_path is None:
            sentiment_path = CACHE_DIR / "daily_sentiment_features.parquet"
        sent = pd.read_parquet(sentiment_path)
        sent["Date"] = pd.to_datetime(sent["Date"])
        df = price.merge(sent, on="Date", how="left")
        # any trading day with no sentiment row -> zeros + flag
        for col in ["finbert_positive", "finbert_negative", "finbert_neutral", "finbert_confidence"]:
            df[col] = df[col].fillna(0.0)
        for col in ["net_sentiment_ma3", "net_sentiment_ma7"]:
            df[col] = df[col].fillna(0.0)
        df["headline_count"] = df["headline_count"].fillna(0).astype(int)
        df["no_news_flag"] = (df["headline_count"] == 0).astype(int)
    else:
        df = price.copy()

    split = chronological_split(df["Date"])
    scalers = fit_scalers(df, split.train, use_sentiment=use_sentiment)
    features = transform_features(df, scalers, use_sentiment=use_sentiment)

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
        "feature_names": ALL_FEATURES if use_sentiment else PRICE_FEATURES,
        "target_cols": target_cols,
    }
