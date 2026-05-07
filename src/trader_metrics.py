"""Trader-facing metrics + the persistence-baseline trap demonstration.

Companion to evaluate.py. Where evaluate.py reports MSE/MAE/DirAcc on the
return space (the loss space the model trains in), this module reports
results in the price space a trader actually thinks in:

  - persistence_baseline:   predicted_close[t+k] = close[t] (the "lying chart")
  - returns_to_prices:      convert return-space predictions to $ predictions
  - hit_rate_at_bands:      fraction of $ predictions inside ±$1, ±$3, ±$5
  - diracc_by_regime:       directional accuracy split by up-day vs down-day
  - strategy_pnl:           long-when-pred>0 vs buy-and-hold, cumulative P&L
  - binomial_diracc_test:   p-value that DirAcc beats baseline (paired McNemar)

Everything operates on the test split. The runner in
scripts/_run_trader_analysis.py wires it together.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import binomtest


HORIZONS = (1, 3, 5)


# ---------------------------------------------------------------------------
# Aligning windowed predictions back to the underlying price series
# ---------------------------------------------------------------------------

@dataclass
class TestAlignment:
    """Rows aligned 1:1 with a model's test predictions.

    end_dates[i]   = date that closes the i-th test window (= the "today" t)
    close_t[i]     = QQQ Close on end_dates[i]
    close_tk[i, j] = QQQ Close on date end_dates[i] + j-th HORIZONS step
    actual_ret[i, j], close_tk - close_t / close_t  (matches y_test exactly)
    """
    end_dates: pd.DatetimeIndex
    close_t: np.ndarray            # (N,)
    close_tk: np.ndarray           # (N, len(HORIZONS))
    actual_ret: np.ndarray         # (N, len(HORIZONS))


def align_test(df: pd.DataFrame, idx_test: np.ndarray, horizons=HORIZONS) -> TestAlignment:
    """Pull Close[t] and Close[t+k] for every test window using its end-index."""
    close = df["Close"].values
    dates = pd.DatetimeIndex(pd.to_datetime(df["Date"].values))
    close_t = close[idx_test]
    close_tk = np.stack([close[idx_test + k] for k in horizons], axis=1)
    actual_ret = close_tk / close_t[:, None] - 1.0
    return TestAlignment(
        end_dates=dates[idx_test],
        close_t=close_t,
        close_tk=close_tk,
        actual_ret=actual_ret,
    )


def returns_to_prices(predicted_returns: np.ndarray, close_t: np.ndarray) -> np.ndarray:
    """Convert a (N, K) matrix of predicted returns to a (N, K) matrix of $ closes."""
    return close_t[:, None] * (1.0 + predicted_returns)


# ---------------------------------------------------------------------------
# Persistence baseline (the "lying chart" trap)
# ---------------------------------------------------------------------------

def persistence_predictions(close_t: np.ndarray, n_horizons: int = 3) -> np.ndarray:
    """predicted_close[t+k] = close[t] for every k. Equivalent to "predicted return = 0"."""
    return np.repeat(close_t[:, None], n_horizons, axis=1)


def persistence_returns(n_rows: int, n_horizons: int = 3) -> np.ndarray:
    """Return-space view of persistence: always 0%."""
    return np.zeros((n_rows, n_horizons), dtype=np.float32)


# ---------------------------------------------------------------------------
# Trader-facing metrics in $ space
# ---------------------------------------------------------------------------

def price_metrics(
    predicted_close: np.ndarray,
    actual_close: np.ndarray,
    horizons=HORIZONS,
    model_name: str = "model",
) -> pd.DataFrame:
    """MSE/MAE/RMSE in dollars, per horizon."""
    rows = []
    for i, k in enumerate(horizons):
        err = predicted_close[:, i] - actual_close[:, i]
        rows.append({
            "model": model_name,
            "horizon": f"t+{k}",
            "price_mse": float(np.mean(err ** 2)),
            "price_mae": float(np.mean(np.abs(err))),
            "price_rmse": float(np.sqrt(np.mean(err ** 2))),
            "n": int(len(err)),
        })
    return pd.DataFrame(rows)


def hit_rate_at_bands(
    predicted_close: np.ndarray,
    actual_close: np.ndarray,
    bands=(1.0, 3.0, 5.0, 10.0),
    horizons=HORIZONS,
    model_name: str = "model",
) -> pd.DataFrame:
    """For each horizon, fraction of |pred - actual| <= each band."""
    rows = []
    for i, k in enumerate(horizons):
        err = np.abs(predicted_close[:, i] - actual_close[:, i])
        row = {"model": model_name, "horizon": f"t+{k}", "n": int(len(err))}
        for b in bands:
            row[f"hit_within_${b:g}"] = float(np.mean(err <= b))
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Directional accuracy split by regime (kills the "period was net-up" objection)
# ---------------------------------------------------------------------------

def diracc_by_regime(
    actual_returns: np.ndarray,
    predicted_returns: np.ndarray,
    horizons=HORIZONS,
    model_name: str = "model",
) -> pd.DataFrame:
    """Directional accuracy on up-days, down-days, and overall.

    A correct prediction = sign(pred) matches sign(actual), with a 0 prediction
    counted as a miss (matches evaluate.per_horizon_metrics convention).
    """
    rows = []
    for i, k in enumerate(horizons):
        yt = actual_returns[:, i]
        yp = predicted_returns[:, i]
        sign_match = (np.sign(yt) == np.sign(yp)) & (yt != 0) & (yp != 0)

        up = yt > 0
        dn = yt < 0
        n_up = int(up.sum())
        n_dn = int(dn.sum())

        rows.append({
            "model": model_name,
            "horizon": f"t+{k}",
            "n": int(len(yt)),
            "n_up_days": n_up,
            "n_down_days": n_dn,
            "diracc_overall": float(sign_match.mean()),
            "diracc_up_days": float(sign_match[up].mean()) if n_up else float("nan"),
            "diracc_down_days": float(sign_match[dn].mean()) if n_dn else float("nan"),
            "balanced_diracc": float(
                (sign_match[up].mean() + sign_match[dn].mean()) / 2.0
            ) if (n_up and n_dn) else float("nan"),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Strategy P&L: long when predicted t+1 return > 0, else cash
# ---------------------------------------------------------------------------

def strategy_pnl(
    predicted_t1_returns: np.ndarray,
    actual_t1_returns: np.ndarray,
    end_dates: pd.DatetimeIndex,
    threshold: float = 0.0,
    cost_per_trade_bps: float = 0.0,
) -> pd.DataFrame:
    """Long-flat strategy on the t+1 return signal.

    Position taken end-of-day t for the day t+1 return.
    Position is 1 if predicted_t1_return > threshold else 0 (cash, 0% return).
    Optional `cost_per_trade_bps` charges a one-way cost on every position change.

    Returns a tidy frame with daily strategy return, cumulative equity, and
    the buy-and-hold equity for direct comparison.
    """
    pos = (predicted_t1_returns > threshold).astype(np.float32)
    raw_strategy = pos * actual_t1_returns

    if cost_per_trade_bps > 0:
        # cost charged when position changes (entry or exit)
        prev_pos = np.concatenate([[0.0], pos[:-1]])
        flips = (pos != prev_pos).astype(np.float32)
        raw_strategy = raw_strategy - flips * (cost_per_trade_bps / 1e4)

    eq_strategy = np.cumprod(1.0 + raw_strategy)
    eq_buyhold = np.cumprod(1.0 + actual_t1_returns)

    return pd.DataFrame({
        "date": end_dates,
        "predicted_ret_t1": predicted_t1_returns,
        "actual_ret_t1": actual_t1_returns,
        "position": pos,
        "strategy_daily_ret": raw_strategy,
        "strategy_equity": eq_strategy,
        "buyhold_equity": eq_buyhold,
    })


def strategy_summary(pnl: pd.DataFrame, label: str) -> dict:
    """One-row summary: total return, ann. Sharpe, max drawdown, exposure."""
    daily = pnl["strategy_daily_ret"].values
    eq = pnl["strategy_equity"].values
    bh_eq = pnl["buyhold_equity"].values
    days = len(daily)

    ann = 252.0
    mu = daily.mean() * ann
    sd = daily.std(ddof=0) * np.sqrt(ann)
    sharpe = float(mu / sd) if sd > 0 else float("nan")

    # max drawdown on the equity curve
    peak = np.maximum.accumulate(eq)
    dd = eq / peak - 1.0
    mdd = float(dd.min())

    return {
        "model": label,
        "n_days": int(days),
        "total_return": float(eq[-1] - 1.0),
        "buyhold_total_return": float(bh_eq[-1] - 1.0),
        "ann_return": float(mu),
        "ann_vol": float(sd),
        "sharpe_ann": sharpe,
        "max_drawdown": mdd,
        "exposure_frac": float(pnl["position"].mean()),
        "n_trades": int((pnl["position"].diff().abs() > 0).sum()),
    }


# ---------------------------------------------------------------------------
# Significance: McNemar paired test on directional accuracy
# ---------------------------------------------------------------------------

def diracc_significance(
    actual_returns: np.ndarray,
    predicted_a: np.ndarray,
    predicted_b: np.ndarray,
    horizons=HORIZONS,
    label_a: str = "baseline",
    label_b: str = "mohamed",
) -> pd.DataFrame:
    """Two paired tests per horizon:

    1. **vs random:** binomial test that model_b's directional accuracy > 0.5.
    2. **vs model_a (McNemar):** of the windows where exactly one of the two
       models was directionally correct, is model_b correct significantly more
       often than model_a? This is the right test for paired predictions on
       the same samples.
    """
    rows = []
    for i, k in enumerate(horizons):
        yt = actual_returns[:, i]
        ca = (np.sign(yt) == np.sign(predicted_a[:, i])) & (yt != 0) & (predicted_a[:, i] != 0)
        cb = (np.sign(yt) == np.sign(predicted_b[:, i])) & (yt != 0) & (predicted_b[:, i] != 0)

        n = int(len(yt))
        n_b_correct = int(cb.sum())
        # binomial vs 0.5
        p_vs_random = float(
            binomtest(n_b_correct, n, p=0.5, alternative="greater").pvalue
        )

        # McNemar's test (exact, two-sided): of the discordant pairs, is b
        # right more often than a?
        b_only = int((cb & ~ca).sum())   # b right, a wrong
        a_only = int((ca & ~cb).sum())   # a right, b wrong
        m = b_only + a_only
        if m == 0:
            p_vs_a = float("nan")
        else:
            # one-sided "b beats a": b_only successes out of m at p=0.5
            p_vs_a = float(
                binomtest(b_only, m, p=0.5, alternative="greater").pvalue
            )

        rows.append({
            "horizon": f"t+{k}",
            "n": n,
            f"{label_a}_correct": int(ca.sum()),
            f"{label_b}_correct": n_b_correct,
            f"{label_b}_diracc": float(cb.mean()),
            f"{label_b}_minus_{label_a}_pp": float(100 * (cb.mean() - ca.mean())),
            "p_vs_random_one_sided": p_vs_random,
            f"mcnemar_{label_b}_only": b_only,
            f"mcnemar_{label_a}_only": a_only,
            "p_mcnemar_b_beats_a": p_vs_a,
        })
    return pd.DataFrame(rows)
