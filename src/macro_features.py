"""V3 macro + technical feature engineering.

Two feature blocks:
  1. Technical features computed from QQQ OHLCV alone (no external data):
     RSI(14), Bollinger %B (20), ATR(14)/Close, OBV z-score, realised
     vol(20), 1-day lag log return, 5-day lag log return, day-of-week.
  2. Macro / cross-asset features fetched once via yfinance and cached:
     VIX, ^TNX (10Y yield), DX-Y.NYB (DXY). All lagged 1 day to avoid
     look-ahead.

Output: results/cache/macro_features_v3.parquet
        - DataFrame keyed on Date with these new columns.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .data_loader import CACHE_DIR, load_qqq

MACRO_PARQUET = CACHE_DIR / "macro_features_v3.parquet"

TECHNICAL_FEATURES = [
    "rsi_14",
    "bollinger_pctb_20",
    "atr_pct_14",
    "obv_z",
    "realised_vol_20",
    "lag_logret_1",
    "lag_logret_5",
    "dow_sin",
    "dow_cos",
]
MACRO_FEATURES = [
    "vix_close",
    "vix_logret",
    "tnx_yield",
    "tnx_diff",
    "dxy_logret",
]
ALL_V3_EXTRA = TECHNICAL_FEATURES + MACRO_FEATURES


# ---------------------------------------------------------------------------
# Technical features (deterministic, OHLCV only)
# ---------------------------------------------------------------------------

def _rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50.0)


def _bollinger_pctb(close: pd.Series, n: int = 20, k: float = 2.0) -> pd.Series:
    ma = close.rolling(n, min_periods=1).mean()
    sd = close.rolling(n, min_periods=1).std().fillna(0)
    upper = ma + k * sd
    lower = ma - k * sd
    width = (upper - lower).replace(0, np.nan)
    return ((close - lower) / width).fillna(0.5).clip(-0.5, 1.5)


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def _obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    direction = np.sign(close.diff().fillna(0))
    return (direction * volume).cumsum()


def _zscore(s: pd.Series, n: int = 60) -> pd.Series:
    mean = s.rolling(n, min_periods=10).mean()
    std = s.rolling(n, min_periods=10).std().replace(0, np.nan)
    return ((s - mean) / std).fillna(0.0).clip(-5, 5)


def add_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add the 9 technical features to a DataFrame with OHLCV columns."""
    out = df.copy()
    close = out["Close"].astype(float)
    high = out["High"].astype(float)
    low = out["Low"].astype(float)
    volume = out["Volume"].astype(float)

    out["rsi_14"] = _rsi(close, 14) / 100.0  # scale into [0, 1]
    out["bollinger_pctb_20"] = _bollinger_pctb(close, 20, 2.0)
    out["atr_pct_14"] = (_atr(high, low, close, 14) / close).fillna(0.0).clip(0, 0.2)
    out["obv_z"] = _zscore(_obv(close, volume), 60)
    logret = np.log(close / close.shift(1)).fillna(0.0)
    out["realised_vol_20"] = logret.rolling(20, min_periods=5).std().fillna(0.0).clip(0, 0.1)
    out["lag_logret_1"] = logret.clip(-0.15, 0.15)
    out["lag_logret_5"] = (np.log(close / close.shift(5))).fillna(0.0).clip(-0.30, 0.30)

    dow = pd.to_datetime(out["Date"]).dt.dayofweek.astype(float)  # Mon=0..Fri=4
    out["dow_sin"] = np.sin(2 * np.pi * dow / 5.0)
    out["dow_cos"] = np.cos(2 * np.pi * dow / 5.0)
    return out


# ---------------------------------------------------------------------------
# Macro / cross-asset (yfinance, cached, lagged 1 day)
# ---------------------------------------------------------------------------

def _fetch_macro(start: str, end: str) -> pd.DataFrame:
    """Fetch VIX / ^TNX / DXY once via yfinance, return a DataFrame keyed on Date."""
    import yfinance as yf

    cols = []
    for ticker, label in [("^VIX", "vix"), ("^TNX", "tnx"), ("DX-Y.NYB", "dxy")]:
        df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        if df.empty:
            print(f"[macro] {ticker} fetch returned empty; will fill with zeros")
            cols.append(pd.DataFrame(index=pd.DatetimeIndex([])).rename_axis("Date"))
            continue
        ser = df[["Close"]].rename(columns={"Close": f"{label}_close_raw"})
        ser.index = pd.to_datetime(ser.index)
        ser.index.name = "Date"
        cols.append(ser)
    out = pd.concat(cols, axis=1).reset_index()
    return out


def add_macro_features(df: pd.DataFrame) -> pd.DataFrame:
    """Merge the lagged macro features onto a DataFrame keyed on Date."""
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    start = df["Date"].min().strftime("%Y-%m-%d")
    end = (df["Date"].max() + pd.Timedelta(days=2)).strftime("%Y-%m-%d")

    macro = _fetch_macro(start, end)
    macro["Date"] = pd.to_datetime(macro["Date"])

    df = df.merge(macro, on="Date", how="left")
    # Forward-fill (handles holiday mismatches between QQQ and yfinance assets)
    for c in ["vix_close_raw", "tnx_close_raw", "dxy_close_raw"]:
        if c in df.columns:
            df[c] = df[c].ffill().bfill()
        else:
            df[c] = np.nan

    # LAG by 1 day to avoid look-ahead - the macro close is published with QQQ's,
    # so for predicting day t we use day t-1's macro values.
    for c in ["vix_close_raw", "tnx_close_raw", "dxy_close_raw"]:
        df[c] = df[c].shift(1)

    df["vix_close"] = df["vix_close_raw"].ffill().fillna(20.0)
    df["vix_logret"] = np.log(df["vix_close_raw"] / df["vix_close_raw"].shift(1)).fillna(0.0).clip(-1, 1)
    df["tnx_yield"] = df["tnx_close_raw"].ffill().fillna(2.0)
    df["tnx_diff"] = df["tnx_close_raw"].diff().fillna(0.0).clip(-1, 1)
    df["dxy_logret"] = np.log(df["dxy_close_raw"] / df["dxy_close_raw"].shift(1)).fillna(0.0).clip(-0.05, 0.05)

    df = df.drop(columns=["vix_close_raw", "tnx_close_raw", "dxy_close_raw"])
    return df


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_v3_macro_cache(force: bool = False) -> Path:
    """Build the v3 technical+macro feature cache once."""
    if MACRO_PARQUET.exists() and not force:
        return MACRO_PARQUET

    qqq = load_qqq(drop_warmup=True)
    qqq = add_technical_features(qqq)
    qqq = add_macro_features(qqq)
    cols = ["Date"] + ALL_V3_EXTRA
    out = qqq[cols]
    MACRO_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(MACRO_PARQUET, index=False)
    return MACRO_PARQUET


if __name__ == "__main__":
    p = build_v3_macro_cache(force=True)
    df = pd.read_parquet(p)
    print(f"wrote {p}  shape={df.shape}")
    print(df.head())
    print(df.describe()[ALL_V3_EXTRA].T[["mean", "std", "min", "max"]].to_string())
