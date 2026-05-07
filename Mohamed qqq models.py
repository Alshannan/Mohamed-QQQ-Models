# Mohamed qqq models.py
#
# Single-file pipeline for the QQQ sentiment-augmented transformer project,
# namely the price-only baseline, my first model v1 (sentiment-fused with an
# LSTM branch), my second model v2 (two-head with a parallel sigmoid direction
# head), and my third model v3 (MANA-Net attention pool plus macro and
# technical features), with a cohort backtest from $10,000.
#
# Run:
#   python "Mohamed qqq models.py" --build-caches   # FinBERT + macro + headlines (one-time)
#   python "Mohamed qqq models.py" --train all      # train all 4 models
#   python "Mohamed qqq models.py" --backtest       # backtest from $10,000
#   python "Mohamed qqq models.py" --all            # everything end to end

import argparse
import math
import os
import sys
import time
from pathlib import Path
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from sklearn.preprocessing import MinMaxScaler, StandardScaler

# ---- paths

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RESULTS = ROOT / "results"
CACHE_DIR = RESULTS / "cache"
MODELS_DIR = RESULTS / "models"
PLOTS_DIR = RESULTS / "plots"
for d in (RESULTS, CACHE_DIR, MODELS_DIR, PLOTS_DIR):
    d.mkdir(parents=True, exist_ok=True)

QQQ_PATH = DATA_DIR / "QQQ_2000_2024_with_MACD.csv"
HEADLINES_PATH = DATA_DIR / "financial_headlines_2000_2024.csv"

PER_HEADLINE_PARQUET = CACHE_DIR / "headline_finbert_scores.parquet"
DAILY_V1_PARQUET = CACHE_DIR / "daily_sentiment_features.parquet"
DAILY_V2_PARQUET = CACHE_DIR / "daily_sentiment_features_v2.parquet"
HEADLINE_NPZ = CACHE_DIR / "per_day_headlines_v3.npz"
MACRO_PARQUET = CACHE_DIR / "macro_features_v3.parquet"

# ---- constants

WARMUP_DROP = 30
HORIZONS = (1, 3, 5)
SEED = 42

TRAIN_END = pd.Timestamp("2017-06-30")
VAL_END = pd.Timestamp("2021-03-31")
TEST_END = pd.Timestamp("2024-12-31")

PRICE_FEATURES = ["Open", "High", "Low", "Close", "Volume", "MACD"]
SENT_FEATURES_V1 = [
    "finbert_positive", "finbert_negative", "finbert_neutral", "finbert_confidence",
    "headline_count", "no_news_flag", "net_sentiment_ma3", "net_sentiment_ma7",
]
SENT_FEATURES_V2 = SENT_FEATURES_V1 + ["confidence_weighted_net_sentiment"]
MACRO_TECH = [
    "rsi_14", "bollinger_pctb_20", "atr_pct_14", "obv_z",
    "realised_vol_20", "lag_logret_1", "lag_logret_5", "dow_sin", "dow_cos",
    "vix_close", "vix_logret", "tnx_yield", "tnx_diff", "dxy_logret",
]
ALL_V3 = PRICE_FEATURES + SENT_FEATURES_V2 + MACRO_TECH

MAX_HEADLINES = 16
SENTIMENT_DIM = 4

FINBERT_MODELS = (
    "ProsusAI/finbert",
    "mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis",
)

# ---- data loading

def load_qqq(drop_warmup=True):
    df = pd.read_csv(QQQ_PATH)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    if drop_warmup:
        df = df.iloc[WARMUP_DROP:].reset_index(drop=True)
    return df


def load_headlines():
    df = pd.read_csv(HEADLINES_PATH)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "headline"]).reset_index(drop=True)
    return df


def add_targets(df, horizons=HORIZONS):
    out = df.copy()
    for k in horizons:
        out[f"target_t{k}"] = out["Close"].shift(-k) / out["Close"] - 1.0
    return out


@dataclass
class Split:
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray


def chrono_split(dates):
    train_mask = dates <= TRAIN_END
    val_mask = (dates > TRAIN_END) & (dates <= VAL_END)
    test_mask = (dates > VAL_END) & (dates <= TEST_END)
    return Split(
        train=np.where(train_mask)[0],
        val=np.where(val_mask)[0],
        test=np.where(test_mask)[0],
    )


# ---- FinBERT scoring (one-time)

def _label_to_idx(label):
    label = label.lower()
    if label in {"positive", "label_2"} or label.startswith("pos"):
        return 0
    if label in {"negative", "label_0"} or label.startswith("neg"):
        return 1
    return 2


def score_headlines(headlines, batch_size=32, max_length=96, progress_every=50):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    last_err = None
    model = tok = used = None
    for name in FINBERT_MODELS:
        try:
            tok = AutoTokenizer.from_pretrained(name)
            model = AutoModelForSequenceClassification.from_pretrained(name)
            used = name
            break
        except Exception as e:
            last_err = e
            print(f"could not load {name}: {e}")
    if model is None:
        raise RuntimeError(f"all FinBERT models failed; last: {last_err}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()
    print(f"finbert: {used} on {device}")

    id2label = {int(i): str(l) for i, l in model.config.id2label.items()}
    perm = [None, None, None]
    for idx, lbl in id2label.items():
        slot = _label_to_idx(lbl)
        perm[slot] = idx
    if any(p is None for p in perm):
        perm = [0, 1, 2]
    perm = np.asarray(perm)

    texts = headlines["headline"].astype(str).tolist()
    n = len(texts)
    probs = np.zeros((n, 3), dtype=np.float32)

    t0 = time.time()
    with torch.no_grad():
        for b in range(0, n, batch_size):
            chunk = texts[b: b + batch_size]
            enc = tok(chunk, padding=True, truncation=True, max_length=max_length, return_tensors="pt").to(device)
            logits = model(**enc).logits
            p = torch.softmax(logits, dim=-1).cpu().numpy()
            probs[b: b + len(chunk)] = p[:, perm]
            if b and (b // batch_size) % progress_every == 0:
                rate = (b + len(chunk)) / max(time.time() - t0, 1e-6)
                eta = (n - b - len(chunk)) / max(rate, 1e-6)
                print(f"  {b + len(chunk)}/{n}  {rate:.1f} hl/s  ETA {eta/60:.1f}min")

    out = headlines[["date", "headline"]].copy()
    out["pos"] = probs[:, 0]
    out["neg"] = probs[:, 1]
    out["neu"] = probs[:, 2]
    out["confidence"] = probs.max(axis=1)
    return out, used


def aggregate_daily_v1(per_hl, trading_days):
    trading_days = pd.Series(sorted(pd.to_datetime(trading_days).unique()))
    hl = per_hl.copy()
    hl["date"] = pd.to_datetime(hl["date"]).dt.normalize()
    pos_idx = trading_days.searchsorted(hl["date"].values, side="left")
    in_range = pos_idx < len(trading_days)
    hl = hl.loc[in_range].copy()
    hl["assigned"] = trading_days.iloc[pos_idx[in_range]].values

    g = hl.groupby("assigned").agg(
        finbert_positive=("pos", "mean"),
        finbert_negative=("neg", "mean"),
        finbert_neutral=("neu", "mean"),
        finbert_confidence=("confidence", "mean"),
        headline_count=("headline", "count"),
    ).reset_index().rename(columns={"assigned": "Date"})

    full = pd.DataFrame({"Date": trading_days})
    daily = full.merge(g, on="Date", how="left")
    fill_zero = ["finbert_positive", "finbert_negative", "finbert_neutral",
                 "finbert_confidence", "headline_count"]
    for c in fill_zero:
        daily[c] = daily[c].fillna(0.0)
    daily["headline_count"] = daily["headline_count"].astype(int)
    daily["no_news_flag"] = (daily["headline_count"] == 0).astype(int)
    daily["net_sentiment"] = daily["finbert_positive"] - daily["finbert_negative"]
    daily["net_sentiment_ma3"] = daily["net_sentiment"].rolling(3, min_periods=1).mean()
    daily["net_sentiment_ma7"] = daily["net_sentiment"].rolling(7, min_periods=1).mean()
    return daily


def add_confidence_weighted_v2(per_hl, trading_days):
    trading_days = pd.Series(sorted(pd.to_datetime(trading_days).unique()))
    hl = per_hl.copy()
    hl["date"] = pd.to_datetime(hl["date"]).dt.normalize()
    pos_idx = trading_days.searchsorted(hl["date"].values, side="left")
    in_range = pos_idx < len(trading_days)
    hl = hl.loc[in_range].copy()
    hl["assigned"] = trading_days.iloc[pos_idx[in_range]].values
    hl["polarity"] = hl["pos"] - hl["neg"]
    hl["w_polarity"] = hl["confidence"] * hl["polarity"]
    g = hl.groupby("assigned").agg(
        wsum=("w_polarity", "sum"),
        csum=("confidence", "sum"),
    ).reset_index().rename(columns={"assigned": "Date"})
    g["confidence_weighted_net_sentiment"] = np.where(
        g["csum"] > 0, g["wsum"] / g["csum"], 0.0
    )
    return g[["Date", "confidence_weighted_net_sentiment"]]


def build_per_day_headline_tensor(per_hl, trading_days):
    trading_days = pd.Series(sorted(pd.to_datetime(trading_days).unique()))
    hl = per_hl.copy()
    hl["date"] = pd.to_datetime(hl["date"]).dt.normalize()
    pos_idx = trading_days.searchsorted(hl["date"].values, side="left")
    in_range = pos_idx < len(trading_days)
    hl = hl.loc[in_range].copy()
    hl["assigned"] = trading_days.iloc[pos_idx[in_range]].values
    hl = hl.sort_values(["assigned", "date"]).reset_index(drop=True)

    n = len(trading_days)
    headlines = np.zeros((n, MAX_HEADLINES, SENTIMENT_DIM), dtype=np.float32)
    mask = np.ones((n, MAX_HEADLINES), dtype=bool)  # True = pad

    day_to_idx = {pd.Timestamp(d): i for i, d in enumerate(trading_days)}
    for assigned, group in hl.groupby("assigned", sort=False):
        i = day_to_idx[pd.Timestamp(assigned)]
        chosen = group.tail(MAX_HEADLINES)
        n_h = len(chosen)
        headlines[i, :n_h, 0] = chosen["pos"].values
        headlines[i, :n_h, 1] = chosen["neg"].values
        headlines[i, :n_h, 2] = chosen["neu"].values
        headlines[i, :n_h, 3] = chosen["confidence"].values
        mask[i, :n_h] = False
    return headlines, mask, np.array(trading_days, dtype="datetime64[ns]")


# ---- macro/technical features (v3)

def _rsi(close, n=14):
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1/n, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/n, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50.0)


def _bollinger_pctb(close, n=20, k=2.0):
    ma = close.rolling(n, min_periods=1).mean()
    sd = close.rolling(n, min_periods=1).std().fillna(0)
    upper = ma + k * sd
    lower = ma - k * sd
    width = (upper - lower).replace(0, np.nan)
    return ((close - lower) / width).fillna(0.5).clip(-0.5, 1.5)


def _atr(high, low, close, n=14):
    prev = close.shift(1)
    tr = pd.concat([(high - low), (high - prev).abs(), (low - prev).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/n, adjust=False).mean()


def _obv(close, vol):
    direction = np.sign(close.diff().fillna(0))
    return (direction * vol).cumsum()


def _zscore(s, n=60):
    mean = s.rolling(n, min_periods=10).mean()
    std = s.rolling(n, min_periods=10).std().replace(0, np.nan)
    return ((s - mean) / std).fillna(0.0).clip(-5, 5)


def add_technical(df):
    out = df.copy()
    close = out["Close"].astype(float)
    high = out["High"].astype(float)
    low = out["Low"].astype(float)
    vol = out["Volume"].astype(float)

    out["rsi_14"] = _rsi(close, 14) / 100.0
    out["bollinger_pctb_20"] = _bollinger_pctb(close, 20, 2.0)
    out["atr_pct_14"] = (_atr(high, low, close, 14) / close).fillna(0.0).clip(0, 0.2)
    out["obv_z"] = _zscore(_obv(close, vol), 60)
    logret = np.log(close / close.shift(1)).fillna(0.0)
    out["realised_vol_20"] = logret.rolling(20, min_periods=5).std().fillna(0.0).clip(0, 0.1)
    out["lag_logret_1"] = logret.clip(-0.15, 0.15)
    out["lag_logret_5"] = (np.log(close / close.shift(5))).fillna(0.0).clip(-0.30, 0.30)

    dow = pd.to_datetime(out["Date"]).dt.dayofweek.astype(float)
    out["dow_sin"] = np.sin(2 * np.pi * dow / 5.0)
    out["dow_cos"] = np.cos(2 * np.pi * dow / 5.0)
    return out


def fetch_macro(start, end):
    import yfinance as yf
    cols = []
    for ticker, label in [("^VIX", "vix"), ("^TNX", "tnx"), ("DX-Y.NYB", "dxy")]:
        df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        if df.empty:
            print(f"  warn: {ticker} returned empty, will fill 0")
            cols.append(pd.DataFrame(index=pd.DatetimeIndex([])).rename_axis("Date"))
            continue
        ser = df[["Close"]].rename(columns={"Close": f"{label}_close_raw"})
        ser.index = pd.to_datetime(ser.index)
        ser.index.name = "Date"
        cols.append(ser)
    return pd.concat(cols, axis=1).reset_index()


def add_macro(df):
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    start = df["Date"].min().strftime("%Y-%m-%d")
    end = (df["Date"].max() + pd.Timedelta(days=2)).strftime("%Y-%m-%d")

    macro = fetch_macro(start, end)
    macro["Date"] = pd.to_datetime(macro["Date"])
    df = df.merge(macro, on="Date", how="left")

    for c in ["vix_close_raw", "tnx_close_raw", "dxy_close_raw"]:
        if c in df.columns:
            df[c] = df[c].ffill().bfill()
        else:
            df[c] = np.nan

    # lag by 1 trading day so today's prediction uses yesterday's macro close
    for c in ["vix_close_raw", "tnx_close_raw", "dxy_close_raw"]:
        df[c] = df[c].shift(1)

    df["vix_close"] = df["vix_close_raw"].ffill().fillna(20.0)
    df["vix_logret"] = np.log(df["vix_close_raw"] / df["vix_close_raw"].shift(1)).fillna(0.0).clip(-1, 1)
    df["tnx_yield"] = df["tnx_close_raw"].ffill().fillna(2.0)
    df["tnx_diff"] = df["tnx_close_raw"].diff().fillna(0.0).clip(-1, 1)
    df["dxy_logret"] = np.log(df["dxy_close_raw"] / df["dxy_close_raw"].shift(1)).fillna(0.0).clip(-0.05, 0.05)
    df = df.drop(columns=["vix_close_raw", "tnx_close_raw", "dxy_close_raw"])
    return df


# ---- cache builders

def build_finbert_cache(force=False):
    if PER_HEADLINE_PARQUET.exists() and not force:
        print(f"already exists: {PER_HEADLINE_PARQUET.name}")
        return PER_HEADLINE_PARQUET
    print("scoring headlines with FinBERT...")
    headlines = load_headlines()
    scored, used = score_headlines(headlines)
    scored.to_parquet(PER_HEADLINE_PARQUET, index=False)
    print(f"saved {PER_HEADLINE_PARQUET}")
    return PER_HEADLINE_PARQUET


def build_v1_daily_cache(force=False):
    if DAILY_V1_PARQUET.exists() and not force:
        return DAILY_V1_PARQUET
    per_hl = pd.read_parquet(PER_HEADLINE_PARQUET)
    qqq = load_qqq(drop_warmup=True)
    daily = aggregate_daily_v1(per_hl, qqq["Date"])
    daily.to_parquet(DAILY_V1_PARQUET, index=False)
    print(f"saved {DAILY_V1_PARQUET}")
    return DAILY_V1_PARQUET


def build_v2_daily_cache(force=False):
    if DAILY_V2_PARQUET.exists() and not force:
        return DAILY_V2_PARQUET
    per_hl = pd.read_parquet(PER_HEADLINE_PARQUET)
    qqq = load_qqq(drop_warmup=True)
    v1 = aggregate_daily_v1(per_hl, qqq["Date"])
    extra = add_confidence_weighted_v2(per_hl, qqq["Date"])
    out = v1.merge(extra, on="Date", how="left")
    out["confidence_weighted_net_sentiment"] = out["confidence_weighted_net_sentiment"].fillna(0.0)
    out.to_parquet(DAILY_V2_PARQUET, index=False)
    print(f"saved {DAILY_V2_PARQUET}")
    return DAILY_V2_PARQUET


def build_headline_npz(force=False):
    if HEADLINE_NPZ.exists() and not force:
        return HEADLINE_NPZ
    per_hl = pd.read_parquet(PER_HEADLINE_PARQUET)
    qqq = load_qqq(drop_warmup=True)
    headlines, mask, dates = build_per_day_headline_tensor(per_hl, qqq["Date"])
    np.savez_compressed(HEADLINE_NPZ, headlines=headlines, mask=mask, dates=dates)
    print(f"saved {HEADLINE_NPZ}")
    return HEADLINE_NPZ


def build_macro_cache(force=False):
    if MACRO_PARQUET.exists() and not force:
        return MACRO_PARQUET
    qqq = load_qqq(drop_warmup=True)
    qqq = add_technical(qqq)
    qqq = add_macro(qqq)
    out = qqq[["Date"] + MACRO_TECH]
    out.to_parquet(MACRO_PARQUET, index=False)
    print(f"saved {MACRO_PARQUET}")
    return MACRO_PARQUET


def build_all_caches(force=False):
    build_finbert_cache(force=force)
    build_v1_daily_cache(force=force)
    build_v2_daily_cache(force=force)
    build_headline_npz(force=force)
    build_macro_cache(force=force)


# ---- dataset assembly

@dataclass
class Scalers:
    ohlcv: MinMaxScaler
    macd: StandardScaler
    headline_count: StandardScaler = None
    vix_mean: float = 0.0
    vix_std: float = 1.0
    tnx_mean: float = 0.0
    tnx_std: float = 1.0


def fit_scalers(df, train_idx, use_sentiment=False, use_macro=False):
    s = Scalers(ohlcv=MinMaxScaler(), macd=StandardScaler())
    s.ohlcv.fit(df.iloc[train_idx][["Open", "High", "Low", "Close", "Volume"]].values)
    s.macd.fit(df.iloc[train_idx][["MACD"]].values)
    if use_sentiment:
        s.headline_count = StandardScaler()
        s.headline_count.fit(np.log1p(df.iloc[train_idx][["headline_count"]].values))
    if use_macro:
        tr = df.iloc[train_idx]
        s.vix_mean, s.vix_std = float(tr["vix_close"].mean()), float(tr["vix_close"].std() + 1e-6)
        s.tnx_mean, s.tnx_std = float(tr["tnx_yield"].mean()), float(tr["tnx_yield"].std() + 1e-6)
    return s


def transform_features(df, scalers, generation):
    """generation: 'baseline' (6), 'v1' (14), 'v2' (15), 'v3' (29)."""
    out = pd.DataFrame(index=df.index)
    ohlcv = scalers.ohlcv.transform(df[["Open", "High", "Low", "Close", "Volume"]].values)
    out["Open"], out["High"], out["Low"], out["Close"], out["Volume"] = ohlcv.T
    out["MACD"] = scalers.macd.transform(df[["MACD"]].values).ravel()

    if generation == "baseline":
        return out[PRICE_FEATURES].values.astype(np.float32)

    # sentiment block (v1 + v2 + v3)
    for c in ["finbert_positive", "finbert_negative", "finbert_neutral",
              "finbert_confidence", "no_news_flag", "net_sentiment_ma3", "net_sentiment_ma7"]:
        out[c] = df[c].values
    out["headline_count"] = scalers.headline_count.transform(
        np.log1p(df[["headline_count"]].values)
    ).ravel()

    if generation == "v1":
        return out[PRICE_FEATURES + SENT_FEATURES_V1].values.astype(np.float32)

    out["confidence_weighted_net_sentiment"] = df["confidence_weighted_net_sentiment"].values
    if generation == "v2":
        return out[PRICE_FEATURES + SENT_FEATURES_V2].values.astype(np.float32)

    # v3 macro/technical (most are already in good ranges)
    for c in MACRO_TECH:
        if c == "vix_close":
            out[c] = ((df[c].values - scalers.vix_mean) / scalers.vix_std).clip(-5, 5)
        elif c == "tnx_yield":
            out[c] = ((df[c].values - scalers.tnx_mean) / scalers.tnx_std).clip(-5, 5)
        else:
            out[c] = df[c].values
    return out[ALL_V3].values.astype(np.float32)


def build_windows(features, targets, window, valid_idx):
    valid_set = set(valid_idx.tolist())
    finite = np.all(np.isfinite(targets), axis=1)
    keep_X, keep_y, keep_idx = [], [], []
    for t in range(window - 1, features.shape[0]):
        if t not in valid_set or not finite[t]:
            continue
        keep_X.append(features[t - window + 1: t + 1])
        keep_y.append(targets[t])
        keep_idx.append(t)
    return (
        np.stack(keep_X).astype(np.float32),
        np.stack(keep_y).astype(np.float32),
        np.array(keep_idx, dtype=np.int64),
    )


def build_headline_windows(headlines, mask, end_idx, window):
    Xh = np.empty((len(end_idx), window, MAX_HEADLINES, SENTIMENT_DIM), dtype=np.float32)
    Xm = np.empty((len(end_idx), window, MAX_HEADLINES), dtype=bool)
    for i, t in enumerate(end_idx):
        Xh[i] = headlines[t - window + 1: t + 1]
        Xm[i] = mask[t - window + 1: t + 1]
    return Xh, Xm


def assemble_dataset(generation):
    """generation: 'baseline'|'v1'|'v2'|'v3'. Returns dict with windows + dates."""
    qqq = load_qqq(drop_warmup=True)
    df = add_targets(qqq, HORIZONS)

    if generation == "baseline":
        window = 30
    else:
        window = 20

    if generation in ("v1", "v2", "v3"):
        sent_path = DAILY_V2_PARQUET if generation in ("v2", "v3") else DAILY_V1_PARQUET
        sent = pd.read_parquet(sent_path)
        sent["Date"] = pd.to_datetime(sent["Date"])
        df = df.merge(sent, on="Date", how="left")
        for c in ["finbert_positive", "finbert_negative", "finbert_neutral",
                  "finbert_confidence", "net_sentiment_ma3", "net_sentiment_ma7"]:
            df[c] = df[c].fillna(0.0)
        if generation in ("v2", "v3"):
            df["confidence_weighted_net_sentiment"] = df["confidence_weighted_net_sentiment"].fillna(0.0)
        df["headline_count"] = df["headline_count"].fillna(0).astype(int)
        df["no_news_flag"] = (df["headline_count"] == 0).astype(int)

    if generation == "v3":
        macro = pd.read_parquet(MACRO_PARQUET)
        macro["Date"] = pd.to_datetime(macro["Date"])
        df = df.merge(macro, on="Date", how="left")
        for c in MACRO_TECH:
            df[c] = df[c].fillna(0.0)

    split = chrono_split(df["Date"])
    scalers = fit_scalers(
        df, split.train,
        use_sentiment=(generation in ("v1", "v2", "v3")),
        use_macro=(generation == "v3"),
    )
    features = transform_features(df, scalers, generation)
    target_cols = [f"target_t{k}" for k in HORIZONS]
    targets = df[target_cols].values.astype(np.float32)

    Xtr, ytr, idxtr = build_windows(features, targets, window, split.train)
    Xva, yva, idxva = build_windows(features, targets, window, split.val)
    Xte, yte, idxte = build_windows(features, targets, window, split.test)

    out = {
        "X_train": Xtr, "y_train": ytr, "idx_train": idxtr,
        "X_val": Xva, "y_val": yva, "idx_val": idxva,
        "X_test": Xte, "y_test": yte, "idx_test": idxte,
        "df": df, "split": split, "scalers": scalers, "window": window,
    }

    if generation == "v3":
        npz = np.load(HEADLINE_NPZ, allow_pickle=False)
        headlines, mask, hl_dates = npz["headlines"], npz["mask"], npz["dates"]
        hl_idx = {pd.Timestamp(d): i for i, d in enumerate(hl_dates)}
        df_dates = pd.to_datetime(df["Date"]).dt.normalize().values
        head_idx = np.array([hl_idx[pd.Timestamp(d)] for d in df_dates], dtype=np.int64)
        H = headlines[head_idx]
        M = mask[head_idx]
        out["H_train"], out["M_train"] = build_headline_windows(H, M, idxtr, window)
        out["H_val"], out["M_val"] = build_headline_windows(H, M, idxva, window)
        out["H_test"], out["M_test"] = build_headline_windows(H, M, idxte, window)

    return out


# ---- models

class BaselineTransformer(nn.Module):
    def __init__(self, n_features=6, seq_len=30, d_model=128, n_heads=2,
                 ffn_dim=128, dropout=0.1, n_horizons=3):
        super().__init__()
        self.proj = nn.Linear(n_features, d_model)
        self.pos_emb = nn.Embedding(seq_len, d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim), nn.ReLU(), nn.Linear(ffn_dim, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, 64), nn.ReLU(), nn.Linear(64, n_horizons),
        )

    def forward(self, x):
        b, t, _ = x.shape
        h = self.proj(x)
        pos = torch.arange(t, device=x.device).unsqueeze(0).expand(b, t)
        h = h + self.pos_emb(pos)
        a, _ = self.attn(h, h, h, need_weights=False)
        h = self.norm1(h + a)
        h = self.norm2(h + self.ffn(h))
        return self.head(h[:, -1, :])


class _SinPE(nn.Module):
    def __init__(self, d_model, max_len=64):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class MohamedModelV1(nn.Module):
    def __init__(self, n_features=14, n_price=6, seq_len=20, d_model=128,
                 n_heads=4, ffn_dim=256, n_layers=2, dropout=0.1,
                 lstm_hidden=64, n_horizons=3):
        super().__init__()
        self.n_price = n_price
        self.proj = nn.Linear(n_features, d_model)
        self.proj_norm = nn.LayerNorm(d_model)
        self.proj_drop = nn.Dropout(dropout)
        self.pos_enc = _SinPE(d_model, seq_len + 8)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=ffn_dim,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.lstm = nn.LSTM(input_size=n_price, hidden_size=lstm_hidden,
                            num_layers=1, batch_first=True)
        head_in = d_model + lstm_hidden
        self.heads = nn.ModuleList([
            nn.Sequential(nn.Linear(head_in, 64), nn.ReLU(), nn.Linear(64, 1))
            for _ in range(n_horizons)
        ])

    def forward(self, x):
        h = self.proj_drop(self.proj_norm(self.proj(x)))
        h = self.pos_enc(h)
        h = self.encoder(h)
        last = h[:, -1, :]
        _, (hN, _) = self.lstm(x[:, :, :self.n_price])
        last = torch.cat([last, hN[-1]], dim=-1)
        return torch.cat([head(last) for head in self.heads], dim=-1)


class MohamedModelV2(nn.Module):
    def __init__(self, n_features=15, n_price=6, seq_len=20, d_model=128,
                 n_heads=4, ffn_dim=256, n_layers=2, dropout=0.1,
                 lstm_hidden=64, n_horizons=3):
        super().__init__()
        self.n_price = n_price
        self.n_horizons = n_horizons
        self.proj = nn.Linear(n_features, d_model)
        self.proj_norm = nn.LayerNorm(d_model)
        self.proj_drop = nn.Dropout(dropout)
        self.pos_enc = _SinPE(d_model, seq_len + 8)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=ffn_dim,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.lstm = nn.LSTM(input_size=n_price, hidden_size=lstm_hidden,
                            num_layers=1, batch_first=True)
        head_in = d_model + lstm_hidden

        self.reg_heads = nn.ModuleList([
            nn.Sequential(nn.Linear(head_in, 64), nn.ReLU(), nn.Linear(64, 1))
            for _ in range(n_horizons)
        ])
        self.dir_heads = nn.ModuleList([
            nn.Sequential(nn.Linear(head_in, 64), nn.ReLU(), nn.Linear(64, 1))
            for _ in range(n_horizons)
        ])

    def forward(self, x):
        h = self.proj_drop(self.proj_norm(self.proj(x)))
        h = self.pos_enc(h)
        h = self.encoder(h)
        last = h[:, -1, :]
        _, (hN, _) = self.lstm(x[:, :, :self.n_price])
        last = torch.cat([last, hN[-1]], dim=-1)
        reg = torch.cat([head(last) for head in self.reg_heads], dim=-1)
        sign_logits = torch.cat([head(last) for head in self.dir_heads], dim=-1)
        return {"reg": reg, "sign_logits": sign_logits}


class _MarketAttnPool(nn.Module):
    """Per-day attention pool over the day's headlines (MANA-Net pattern)."""
    NEG_INF = -1e9

    def __init__(self, base_dim, sentiment_dim=4, d_model=32, n_heads=2,
                 diff_enlarge=1.5):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.diff = diff_enlarge
        self.q_proj = nn.Linear(base_dim, d_model, bias=False)
        self.k_proj = nn.Linear(sentiment_dim, d_model, bias=False)
        self.v_proj = nn.Linear(sentiment_dim, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, base, hl, mask):
        # base: (B, T, F), hl: (B, T, H, S), mask: (B, T, H) - True=pad
        B, T, H, S = hl.shape
        Q = self.q_proj(base).view(B * T, self.n_heads, self.head_dim)
        K = self.k_proj(hl).view(B * T, H, self.n_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(hl).view(B * T, H, self.n_heads, self.head_dim).transpose(1, 2)
        flat_mask = mask.view(B * T, H)

        scores = torch.einsum("bnd,bnhd->bnh", Q, K) / math.sqrt(self.head_dim)
        scores = scores * self.diff
        scores = scores.masked_fill(flat_mask.unsqueeze(1), self.NEG_INF)
        all_pad = flat_mask.all(dim=-1, keepdim=True)
        attn = torch.softmax(scores, dim=-1)
        attn = attn.masked_fill(all_pad.unsqueeze(1), 0.0)
        pooled = torch.einsum("bnh,bnhd->bnd", attn, V).contiguous().view(B * T, self.d_model)
        return self.out_proj(pooled).view(B, T, self.d_model)


class MohamedModelV3(nn.Module):
    def __init__(self, n_features=29, n_price=6, sentiment_dim=4, max_headlines=16,
                 seq_len=20, d_model=128, n_heads=4, ffn_dim=256, n_layers=2,
                 dropout=0.1, lstm_hidden=64, attn_pool_dim=32,
                 n_horizons=3, n_classes=5):
        super().__init__()
        self.n_price = n_price
        self.n_horizons = n_horizons
        self.n_classes = n_classes

        self.attn_pool = _MarketAttnPool(
            base_dim=n_features, sentiment_dim=sentiment_dim,
            d_model=attn_pool_dim, n_heads=2,
        )

        proj_in = n_features + attn_pool_dim
        self.proj = nn.Linear(proj_in, d_model)
        self.proj_norm = nn.LayerNorm(d_model)
        self.proj_drop = nn.Dropout(dropout)
        self.pos_enc = _SinPE(d_model, seq_len + 8)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=ffn_dim,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.lstm = nn.LSTM(input_size=n_price, hidden_size=lstm_hidden,
                            num_layers=1, batch_first=True)
        head_in = d_model + lstm_hidden

        self.reg_heads = nn.ModuleList([
            nn.Sequential(nn.Linear(head_in, 64), nn.ReLU(), nn.Linear(64, 1))
            for _ in range(n_horizons)
        ])
        self.cls_heads = nn.ModuleList([
            nn.Sequential(nn.Linear(head_in, 64), nn.ReLU(), nn.Linear(64, n_classes))
            for _ in range(n_horizons)
        ])

    def forward(self, x, hl, mask):
        pooled = self.attn_pool(x, hl, mask)
        h = torch.cat([x, pooled], dim=-1)
        h = self.proj_drop(self.proj_norm(self.proj(h)))
        h = self.pos_enc(h)
        h = self.encoder(h)
        last = h[:, -1, :]
        _, (hN, _) = self.lstm(x[:, :, :self.n_price])
        last = torch.cat([last, hN[-1]], dim=-1)
        reg = torch.cat([head(last) for head in self.reg_heads], dim=-1)
        cls = torch.stack([head(last) for head in self.cls_heads], dim=1)
        return {"reg": reg, "cls_logits": cls}


# ---- training

def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_loaders(X_train, y_train, X_val, y_val, bs, balanced=False, *extras_train, **kw):
    if extras_train:
        train_tensors = [torch.from_numpy(X_train)] + [torch.from_numpy(t) for t in extras_train] + [torch.from_numpy(y_train)]
        val_extras = kw.get("val_extras", ())
        val_tensors = [torch.from_numpy(X_val)] + [torch.from_numpy(t) for t in val_extras] + [torch.from_numpy(y_val)]
        train_ds = TensorDataset(*train_tensors)
        val_ds = TensorDataset(*val_tensors)
    else:
        train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
        val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))

    if balanced:
        sign = np.sign(y_train[:, 0])
        sign[sign == 0] = 1
        classes, counts = np.unique(sign, return_counts=True)
        inv = {c: 1.0 / f for c, f in zip(classes.tolist(), counts.tolist())}
        weights = np.array([inv[s] for s in sign.tolist()], dtype=np.float64)
        sampler = WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)
        train_dl = DataLoader(train_ds, batch_size=bs, sampler=sampler)
    else:
        train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=bs * 4, shuffle=False)
    return train_dl, val_dl


def train_simple_mse(model, X_tr, y_tr, X_va, y_va, save_path,
                      epochs=60, bs=64, lr=1e-3, wd=1e-5, patience=8,
                      grad_clip=1.0, seed=42):
    """Used by baseline (lr=1e-3) and v1 (lr=5e-4)."""
    set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    train_dl, val_dl = make_loaders(X_tr, y_tr, X_va, y_va, bs)
    loss_fn = nn.MSELoss()

    best = float("inf")
    best_state = None
    since = 0
    log = []
    t_start = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        tr_loss = 0.0; n_tr = 0
        t0 = time.time()
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            preds = model(xb)
            loss = loss_fn(preds, yb)
            opt.zero_grad(); loss.backward()
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
            tr_loss += loss.item() * xb.size(0); n_tr += xb.size(0)
        tr_loss /= max(n_tr, 1)

        model.eval()
        va_loss = 0.0; n_va = 0
        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(device), yb.to(device)
                preds = model(xb)
                loss = loss_fn(preds, yb)
                va_loss += loss.item() * xb.size(0); n_va += xb.size(0)
        va_loss /= max(n_va, 1)
        dt = time.time() - t0

        improved = va_loss < best
        if improved:
            best = va_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            since = 0
        else:
            since += 1
        log.append({"epoch": ep, "train_loss": tr_loss, "val_loss": va_loss, "seconds": dt})
        flag = " *" if improved else ""
        print(f"  ep {ep:02d}  train {tr_loss:.6f}  val {va_loss:.6f}  ({dt:.1f}s){flag}")
        if since >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
        torch.save({"state_dict": best_state, "best_val_loss": best}, save_path)
    print(f"  saved {save_path}  best val {best:.6f}  total {time.time() - t_start:.1f}s")
    return model, log


def train_v2(model, X_tr, y_tr, X_va, y_va, save_path,
             epochs=60, bs=64, lr=5e-4, wd=1e-5, patience=8,
             grad_clip=1.0, alpha=1.0, seed=42):
    set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    norm_mse = float(np.var(y_tr, ddof=0).mean())
    train_dl, val_dl = make_loaders(X_tr, y_tr, X_va, y_va, bs, balanced=True)
    bce = nn.BCEWithLogitsLoss()

    best = float("inf")
    best_state = None
    since = 0
    log = []
    t_start = time.time()
    for ep in range(1, epochs + 1):
        for split_label, dl, training in [("train", train_dl, True), ("val", val_dl, False)]:
            model.train(training)
            total = 0.0; n = 0
            t0 = time.time()
            ctx = torch.enable_grad() if training else torch.no_grad()
            with ctx:
                for xb, yb in dl:
                    xb, yb = xb.to(device), yb.to(device)
                    out = model(xb)
                    reg_loss = F.mse_loss(out["reg"], yb) / norm_mse
                    sign_target = (yb > 0).float()
                    bce_loss = bce(out["sign_logits"], sign_target)
                    loss = reg_loss + alpha * bce_loss
                    if training:
                        opt.zero_grad(); loss.backward()
                        if grad_clip:
                            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                        opt.step()
                    total += loss.item() * xb.size(0); n += xb.size(0)
            avg = total / max(n, 1)
            if split_label == "train":
                tr_loss, dt = avg, time.time() - t0
            else:
                va_loss = avg

        improved = va_loss < best
        if improved:
            best = va_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            since = 0
        else:
            since += 1
        log.append({"epoch": ep, "train_loss": tr_loss, "val_loss": va_loss, "epoch_seconds": dt, "improved": int(improved)})
        flag = " *" if improved else ""
        print(f"  ep {ep:02d}  train {tr_loss:.6f}  val {va_loss:.6f}  ({dt:.1f}s){flag}")
        if since >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
        torch.save({"state_dict": best_state, "best_val_loss": best}, save_path)
    print(f"  saved {save_path}  best val {best:.6f}  total {time.time() - t_start:.1f}s")
    return model, log


CLS_THRESHOLDS = np.array([-0.03, -0.01, 0.01, 0.03], dtype=np.float32)


def returns_to_classes(y):
    return np.searchsorted(CLS_THRESHOLDS, y, side="right").astype(np.int64)


def train_v3(model, X_tr, H_tr, M_tr, y_tr, X_va, H_va, M_va, y_va, save_path,
             epochs=80, bs=64, lr=3e-4, wd=1e-5, patience=10,
             grad_clip=1.0, alpha=0.0, seed=42):
    """alpha=0 means cls head is created but not trained (we found this works
    best for v3; other alphas in {0.1, 0.3, 1.0} were tried)."""
    set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    norm_mse = float(np.var(y_tr, ddof=0).mean())
    rcap_mean = float(np.clip(np.abs(y_tr), 0, 0.5).mean()) + 1e-6

    n_h = y_tr.shape[1]
    weights = np.zeros((n_h, len(CLS_THRESHOLDS) + 1), dtype=np.float32)
    for h in range(n_h):
        cls = returns_to_classes(y_tr[:, h])
        cnt = np.bincount(cls, minlength=len(CLS_THRESHOLDS) + 1) + 1
        inv = np.sqrt(1.0 / cnt)
        weights[h] = inv / inv.mean()
    cls_weights = torch.tensor(weights, dtype=torch.float32, device=device)

    train_ds = TensorDataset(
        torch.from_numpy(X_tr), torch.from_numpy(H_tr),
        torch.from_numpy(M_tr), torch.from_numpy(y_tr),
    )
    val_ds = TensorDataset(
        torch.from_numpy(X_va), torch.from_numpy(H_va),
        torch.from_numpy(M_va), torch.from_numpy(y_va),
    )

    sign = np.sign(y_tr[:, 0]); sign[sign == 0] = 1
    classes, counts = np.unique(sign, return_counts=True)
    inv = {c: 1.0 / f for c, f in zip(classes.tolist(), counts.tolist())}
    sample_w = np.array([inv[s] for s in sign.tolist()], dtype=np.float64)
    sampler = WeightedRandomSampler(weights=sample_w, num_samples=len(sample_w), replacement=True)
    train_dl = DataLoader(train_ds, batch_size=bs, sampler=sampler)
    val_dl = DataLoader(val_ds, batch_size=bs * 4, shuffle=False)

    best = float("inf")
    best_state = None
    since = 0
    log = []
    t_start = time.time()
    for ep in range(1, epochs + 1):
        for split_label, dl, training in [("train", train_dl, True), ("val", val_dl, False)]:
            model.train(training)
            total = 0.0; n = 0
            t0 = time.time()
            ctx = torch.enable_grad() if training else torch.no_grad()
            with ctx:
                for xb, hb, mb, yb in dl:
                    xb, hb, mb, yb = xb.to(device), hb.to(device), mb.to(device), yb.to(device)
                    out = model(xb, hb, mb)
                    reg_loss = F.mse_loss(out["reg"], yb) / norm_mse

                    if alpha > 0:
                        r_cap = torch.clamp(yb, -0.5, 0.5).abs()
                        cls_target = torch.zeros_like(yb, dtype=torch.long)
                        for t in CLS_THRESHOLDS:
                            cls_target = cls_target + (yb >= t).long()
                        ce_per = torch.zeros_like(yb)
                        for h in range(n_h):
                            ce_per[:, h] = F.cross_entropy(
                                out["cls_logits"][:, h, :], cls_target[:, h],
                                weight=cls_weights[h], reduction="none",
                            )
                        cls_loss = (ce_per * (r_cap / rcap_mean)).mean()
                        loss = reg_loss + alpha * cls_loss
                    else:
                        loss = reg_loss

                    if training:
                        opt.zero_grad(); loss.backward()
                        if grad_clip:
                            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                        opt.step()
                    total += loss.item() * xb.size(0); n += xb.size(0)
            avg = total / max(n, 1)
            if split_label == "train":
                tr_loss, dt = avg, time.time() - t0
            else:
                va_loss = avg

        improved = va_loss < best
        if improved:
            best = va_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            since = 0
        else:
            since += 1
        log.append({"epoch": ep, "train_loss": tr_loss, "val_loss": va_loss, "epoch_seconds": dt, "improved": int(improved)})
        flag = " *" if improved else ""
        print(f"  ep {ep:02d}  train {tr_loss:.6f}  val {va_loss:.6f}  ({dt:.1f}s){flag}")
        if since >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
        torch.save({"state_dict": best_state, "best_val_loss": best}, save_path)
    print(f"  saved {save_path}  best val {best:.6f}  total {time.time() - t_start:.1f}s")
    return model, log


# ---- prediction

@torch.no_grad()
def predict(model, X, batch_size=256):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()
    out = []
    for b in range(0, len(X), batch_size):
        xb = torch.from_numpy(X[b: b + batch_size]).to(device)
        out.append(model(xb).cpu().numpy())
    return np.concatenate(out, axis=0)


@torch.no_grad()
def predict_v2(model, X, batch_size=256):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()
    reg, sig = [], []
    for b in range(0, len(X), batch_size):
        xb = torch.from_numpy(X[b: b + batch_size]).to(device)
        o = model(xb)
        reg.append(o["reg"].cpu().numpy())
        sig.append(torch.sigmoid(o["sign_logits"]).cpu().numpy())
    return {"reg": np.concatenate(reg, axis=0), "sign_prob": np.concatenate(sig, axis=0)}


@torch.no_grad()
def predict_v3(model, X, H, M, batch_size=128):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()
    reg, cls = [], []
    for b in range(0, len(X), batch_size):
        xb = torch.from_numpy(X[b: b + batch_size]).to(device)
        hb = torch.from_numpy(H[b: b + batch_size]).to(device)
        mb = torch.from_numpy(M[b: b + batch_size]).to(device)
        o = model(xb, hb, mb)
        reg.append(o["reg"].cpu().numpy())
        cls.append(F.softmax(o["cls_logits"], dim=-1).cpu().numpy())
    return {"reg": np.concatenate(reg, axis=0), "cls_prob": np.concatenate(cls, axis=0)}


# ---- training entry points

def train_baseline():
    print("=== baseline ===")
    data = assemble_dataset("baseline")
    print(f"  X_train {data['X_train'].shape}  X_val {data['X_val'].shape}  X_test {data['X_test'].shape}")
    m = BaselineTransformer()
    n_params = sum(p.numel() for p in m.parameters())
    print(f"  params: {n_params:,}")
    train_simple_mse(
        m, data["X_train"], data["y_train"], data["X_val"], data["y_val"],
        save_path=MODELS_DIR / "baseline.pt",
        epochs=60, lr=1e-3, patience=8, seed=SEED,
    )


def train_v1():
    print("=== mohamed v1 ===")
    data = assemble_dataset("v1")
    print(f"  X_train {data['X_train'].shape}  X_val {data['X_val'].shape}  X_test {data['X_test'].shape}")
    m = MohamedModelV1()
    n_params = sum(p.numel() for p in m.parameters())
    print(f"  params: {n_params:,}")
    train_simple_mse(
        m, data["X_train"], data["y_train"], data["X_val"], data["y_val"],
        save_path=MODELS_DIR / "mohamed.pt",
        epochs=60, lr=5e-4, patience=8, seed=SEED,
    )


def train_v2_runner():
    print("=== mohamed v2 ===")
    data = assemble_dataset("v2")
    print(f"  X_train {data['X_train'].shape}  X_val {data['X_val'].shape}  X_test {data['X_test'].shape}")
    m = MohamedModelV2()
    n_params = sum(p.numel() for p in m.parameters())
    print(f"  params: {n_params:,}")
    train_v2(
        m, data["X_train"], data["y_train"], data["X_val"], data["y_val"],
        save_path=MODELS_DIR / "mohamed_v2.pt",
        epochs=60, lr=5e-4, patience=8, alpha=1.0, seed=SEED,
    )


def train_v3_runner():
    print("=== mohamed v3 ===")
    data = assemble_dataset("v3")
    print(f"  X_train {data['X_train'].shape}  H_train {data['H_train'].shape}  X_test {data['X_test'].shape}")
    n_features = data["X_train"].shape[-1]
    m = MohamedModelV3(n_features=n_features)
    n_params = sum(p.numel() for p in m.parameters())
    print(f"  params: {n_params:,}")
    train_v3(
        m, data["X_train"], data["H_train"], data["M_train"], data["y_train"],
        data["X_val"], data["H_val"], data["M_val"], data["y_val"],
        save_path=MODELS_DIR / "mohamed_v3.pt",
        epochs=80, lr=3e-4, patience=10, alpha=0.0, seed=SEED,
    )


# ---- backtest

START_CAPITAL = 10_000.0
COST_BPS = 5.0


def cohort_backtest(predicted_k, actual_k, dates, k,
                    capital=START_CAPITAL, cost_bps=COST_BPS):
    n = len(predicted_k)
    cost = cost_bps / 1e4
    rows = [{"date": dates[0], "position": 0, "realised_ret": 0.0,
             "equity": capital, "trade": 0}]
    equity = capital
    prev_pos = 0
    t = 0
    while t + k < n:
        target = 1 if predicted_k[t] > 0 else 0
        flipped = int(target != prev_pos)
        if flipped:
            equity = equity * (1.0 - cost)
        if target == 1:
            equity = equity * (1.0 + float(actual_k[t]))
        rows.append({
            "date": dates[t + k], "position": target,
            "realised_ret": float(actual_k[t]) if target == 1 else 0.0,
            "equity": equity, "trade": flipped,
        })
        prev_pos = target
        t += k
    return pd.DataFrame(rows)


def buyhold_cohort(actual_k, dates, k, capital=START_CAPITAL):
    n = len(actual_k)
    rows = [{"date": dates[0], "position": 1, "realised_ret": 0.0,
             "equity": capital, "trade": 0}]
    equity = capital
    t = 0
    while t + k < n:
        equity = equity * (1.0 + float(actual_k[t]))
        rows.append({"date": dates[t + k], "position": 1,
                     "realised_ret": float(actual_k[t]),
                     "equity": equity, "trade": 0})
        t += k
    return pd.DataFrame(rows)


def summarise_strategy(df, model, k, start=START_CAPITAL):
    eq = df["equity"].values
    decision_rets = df["equity"].pct_change().fillna(0.0).values[1:]
    if len(decision_rets) == 0 or decision_rets.std(ddof=0) == 0:
        sharpe = 0.0
    else:
        sharpe = float(decision_rets.mean() / decision_rets.std(ddof=0) * np.sqrt(252.0 / k))
    peak = np.maximum.accumulate(eq)
    dd = float((eq / peak - 1.0).min()) * 100
    return {
        "model": model, "horizon": f"t+{k}",
        "final_dollars": float(eq[-1]),
        "total_return_pct": (float(eq[-1]) / start - 1.0) * 100,
        "sharpe_ann": sharpe,
        "max_drawdown_pct": dd,
        "exposure": float(df["position"].iloc[1:].mean()) if len(df) > 1 else 0.0,
        "n_decisions": len(df) - 1,
        "n_trades": int(df["trade"].sum()),
    }


def run_backtest():
    print("=== backtest from $10,000 ===")
    base_d = assemble_dataset("baseline")
    v1_d = assemble_dataset("v1")
    v2_d = assemble_dataset("v2")
    v3_d = assemble_dataset("v3")

    # load checkpoints
    def _load(model, path):
        blob = torch.load(path, map_location="cpu", weights_only=False)
        state = blob["state_dict"] if isinstance(blob, dict) and "state_dict" in blob else blob
        model.load_state_dict(state); model.eval()
        return model

    base = _load(BaselineTransformer(), MODELS_DIR / "baseline.pt")
    v1 = _load(MohamedModelV1(), MODELS_DIR / "mohamed.pt")
    v2 = _load(MohamedModelV2(), MODELS_DIR / "mohamed_v2.pt")
    v3 = _load(MohamedModelV3(n_features=v3_d["X_train"].shape[-1]), MODELS_DIR / "mohamed_v3.pt")

    base_p = predict(base, base_d["X_test"])
    v1_p = predict(v1, v1_d["X_test"])
    v2_p = predict_v2(v2, v2_d["X_test"])["reg"]
    v3_p = predict_v3(v3, v3_d["X_test"], v3_d["H_test"], v3_d["M_test"])["reg"]

    df_v3 = v3_d["df"]
    test_dates = pd.DatetimeIndex(pd.to_datetime(df_v3["Date"].values[v3_d["idx_test"]]))
    actual = v3_d["y_test"]
    n = len(actual)

    if not (len(base_p) == n and len(v1_p) == n and len(v2_p) == n and len(v3_p) == n):
        # this can happen if window lengths produce different test counts
        # (baseline uses window=30, others use window=20). Crop to the
        # shortest series and align dates with v3.
        m_min = min(len(base_p), len(v1_p), len(v2_p), len(v3_p), n)
        offset = n - m_min
        actual = actual[offset:]
        test_dates = test_dates[offset:]
        if len(base_p) > m_min: base_p = base_p[-m_min:]
        if len(v1_p) > m_min: v1_p = v1_p[-m_min:]
        if len(v2_p) > m_min: v2_p = v2_p[-m_min:]
        if len(v3_p) > m_min: v3_p = v3_p[-m_min:]

    print(f"  {n} test windows  {test_dates[0].date()} to {test_dates[-1].date()}")

    summaries = []
    long_eq = []

    models = [
        ("baseline", base_p),
        ("mohamed_v1", v1_p),
        ("mohamed_v2", v2_p),
        ("mohamed_v3", v3_p),
    ]

    for k_idx, k in enumerate(HORIZONS):
        bh = buyhold_cohort(actual[:, k_idx], test_dates, k)
        summaries.append(summarise_strategy(bh, "buy_and_hold", k))
        for _, row in bh.iterrows():
            long_eq.append({"horizon": f"t+{k}", "model": "buy_and_hold",
                            "date": row["date"], "equity": row["equity"]})

        for name, pred in models:
            df = cohort_backtest(pred[:, k_idx], actual[:, k_idx], test_dates, k)
            summaries.append(summarise_strategy(df, name, k))
            for _, row in df.iterrows():
                long_eq.append({"horizon": f"t+{k}", "model": name,
                                "date": row["date"], "equity": row["equity"]})

    summary = pd.DataFrame(summaries)
    summary.to_csv(RESULTS / "backtest_summary.csv", index=False)
    long_eq_df = pd.DataFrame(long_eq)
    long_eq_df.to_csv(RESULTS / "backtest_equity_long.csv", index=False)

    # plots
    colors = {
        "buy_and_hold": "black", "baseline": "tab:gray",
        "mohamed_v1": "tab:orange", "mohamed_v2": "tab:purple",
        "mohamed_v3": "tab:blue",
    }
    labels = {
        "buy_and_hold": "Buy and Hold", "baseline": "Baseline",
        "mohamed_v1": "Mohamed v1", "mohamed_v2": "Mohamed v2",
        "mohamed_v3": "Mohamed v3",
    }

    for k in HORIZONS:
        fig, ax = plt.subplots(figsize=(10, 5))
        sub = long_eq_df[long_eq_df["horizon"] == f"t+{k}"]
        for m in ["buy_and_hold", "baseline", "mohamed_v1", "mohamed_v2", "mohamed_v3"]:
            cur = sub[sub["model"] == m].sort_values("date")
            final = float(cur["equity"].iloc[-1])
            ax.plot(cur["date"], cur["equity"], lw=1.6,
                    color=colors[m], label=f"{labels[m]} (${final:,.0f})")
        ax.axhline(START_CAPITAL, color="grey", ls=":", lw=0.8)
        ax.set_title(f"Backtest at t+{k} from $10,000 (5 bps round-trip)")
        ax.set_xlabel("date"); ax.set_ylabel("portfolio value ($)")
        ax.legend(loc="upper left", fontsize=10)
        fig.tight_layout()
        fig.savefig(PLOTS_DIR / f"backtest_t{k}.png", dpi=130)
        plt.close(fig)

    # bar chart of finals
    models_order = ["buy_and_hold", "baseline", "mohamed_v1", "mohamed_v2", "mohamed_v3"]
    horizons_str = [f"t+{k}" for k in HORIZONS]
    bar_data = np.zeros((len(models_order), len(horizons_str)))
    for i, m in enumerate(models_order):
        for j, h in enumerate(horizons_str):
            r = summary[(summary["model"] == m) & (summary["horizon"] == h)]
            bar_data[i, j] = float(r["final_dollars"].iloc[0])

    fig, ax = plt.subplots(figsize=(11, 5.5))
    x = np.arange(len(horizons_str))
    width = 0.16
    for i, m in enumerate(models_order):
        offset = (i - (len(models_order) - 1) / 2) * width
        ax.bar(x + offset, bar_data[i], width, color=colors[m], label=labels[m])
        for j, v in enumerate(bar_data[i]):
            ax.text(x[j] + offset, v + 200, f"${v:,.0f}", ha="center", va="bottom", fontsize=8)
    ax.axhline(START_CAPITAL, color="grey", ls=":", lw=0.8, label="start ($10,000)")
    ax.set_xticks(x); ax.set_xticklabels(horizons_str)
    ax.set_ylabel("final portfolio value ($)")
    ax.set_title("Final $ from $10,000 by model and horizon")
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "backtest_finals.png", dpi=130)
    plt.close(fig)

    # print summary
    print("")
    print("=" * 72)
    print("BACKTEST COMPLETE  ·  $10,000 starting capital  ·  5 bps round-trip cost")
    print("=" * 72)
    print(f"{'':<14}{'t+1':>10}{'t+3':>10}{'t+5':>10}")
    for m in models_order:
        row = f"{labels[m]:<14}"
        for h in horizons_str:
            v = summary[(summary["model"] == m) & (summary["horizon"] == h)]["final_dollars"].iloc[0]
            row += f"{f'${v:,.0f}':>10}"
        print(row)
    print("=" * 72)
    print(f"Plots saved to {PLOTS_DIR}")
    print(f"Summary saved to {RESULTS / 'backtest_summary.csv'}")
    print("")


# ---- CLI

def main():
    ap = argparse.ArgumentParser(description="QQQ sentiment-augmented transformer pipeline")
    ap.add_argument("--build-caches", action="store_true",
                    help="run FinBERT inference + build daily/headline/macro caches (one-time)")
    ap.add_argument("--force-cache", action="store_true",
                    help="rebuild caches even if they exist")
    ap.add_argument("--train", choices=["baseline", "v1", "v2", "v3", "all"],
                    help="train one model or all four")
    ap.add_argument("--backtest", action="store_true",
                    help="run cohort backtest from $10,000")
    ap.add_argument("--all", action="store_true",
                    help="build caches, train all models, run backtest")
    args = ap.parse_args()

    if not any([args.build_caches, args.train, args.backtest, args.all]):
        ap.print_help()
        return 0

    if args.all:
        build_all_caches(force=args.force_cache)
        train_baseline()
        train_v1()
        train_v2_runner()
        train_v3_runner()
        run_backtest()
        return 0

    if args.build_caches:
        build_all_caches(force=args.force_cache)

    if args.train == "baseline":
        train_baseline()
    elif args.train == "v1":
        train_v1()
    elif args.train == "v2":
        train_v2_runner()
    elif args.train == "v3":
        train_v3_runner()
    elif args.train == "all":
        train_baseline()
        train_v1()
        train_v2_runner()
        train_v3_runner()

    if args.backtest:
        run_backtest()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
