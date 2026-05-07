"""Metric helpers consumed by the evaluation notebook.

No plotting here - notebooks render figures.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def per_horizon_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    horizons=(1, 3, 5),
    model_name: str = "model",
) -> pd.DataFrame:
    """Return a tidy DataFrame with MSE / MAE / DirAcc per horizon."""
    rows = []
    for i, k in enumerate(horizons):
        yt = y_true[:, i]
        yp = y_pred[:, i]
        mse = float(np.mean((yt - yp) ** 2))
        mae = float(np.mean(np.abs(yt - yp)))
        sign_match = (np.sign(yt) == np.sign(yp))
        # Treat sign(0)=0 as a miss to be conservative.
        dir_acc = float(np.mean(sign_match & (yt != 0)))
        rows.append({
            "model": model_name,
            "horizon": f"t+{k}",
            "mse": mse,
            "mae": mae,
            "directional_accuracy": dir_acc,
            "n": int(len(yt)),
        })
    return pd.DataFrame(rows)


def residuals(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Per-sample, per-horizon residual = predicted - actual."""
    return y_pred - y_true


def worst_predictions(
    df_dates: pd.Series,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    horizon_idx: int = 0,
    top_k: int = 10,
    headlines_by_date: dict | None = None,
) -> pd.DataFrame:
    """Return the top-K largest |error| rows for a chosen horizon."""
    err = np.abs(y_pred[:, horizon_idx] - y_true[:, horizon_idx])
    order = np.argsort(-err)[:top_k]
    out = pd.DataFrame({
        "date": pd.to_datetime(df_dates.values[order]),
        "actual": y_true[order, horizon_idx],
        "predicted": y_pred[order, horizon_idx],
        "abs_error": err[order],
    })
    if headlines_by_date is not None:
        out["headlines"] = out["date"].map(
            lambda d: " | ".join(headlines_by_date.get(pd.Timestamp(d).normalize(), []))
        )
    return out


def permutation_importance(
    predict_fn,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    n_repeats: int = 3,
    rng_seed: int = 42,
) -> pd.DataFrame:
    """Permutation importance: shuffle each feature across the time axis and
    measure the increase in MSE. Higher = the model relies on this feature."""
    rng = np.random.default_rng(rng_seed)
    base = predict_fn(X)
    baseline_mse = float(np.mean((base - y) ** 2))
    rows = []
    for i, name in enumerate(feature_names):
        deltas = []
        for _ in range(n_repeats):
            X_perm = X.copy()
            # Shuffle column i across all (sample, time) positions
            flat = X_perm[:, :, i].reshape(-1).copy()
            rng.shuffle(flat)
            X_perm[:, :, i] = flat.reshape(X_perm.shape[0], X_perm.shape[1])
            preds = predict_fn(X_perm)
            mse = float(np.mean((preds - y) ** 2))
            deltas.append(mse - baseline_mse)
        rows.append({
            "feature": name,
            "delta_mse_mean": float(np.mean(deltas)),
            "delta_mse_std": float(np.std(deltas)),
        })
    return pd.DataFrame(rows).sort_values("delta_mse_mean", ascending=False).reset_index(drop=True)
