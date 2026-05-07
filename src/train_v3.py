"""V3 training loop with combined regression + return-weighted classification loss.

Loss:
  total = MSE / var(y_train)
        + alpha * mean_h( CE(cls_logits_h, cls_target_h) * |r_cap_h| / mean(|r_cap_train|) )

  cls_target is a 5-class bin of the actual return at thresholds
  [-3%, -1%, +1%, +3%]. r_cap is the same return clipped to [-0.5, 0.5]
  to bound the loss weight.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from .train import set_seed

# Class boundaries: returns are binned into 5 buckets
# 0: r < -3%   1: -3% <= r < -1%   2: -1% <= r < +1%   3: +1% <= r < +3%   4: r >= +3%
CLASS_THRESHOLDS = np.array([-0.03, -0.01, 0.01, 0.03], dtype=np.float32)
N_CLASSES = 5


def returns_to_classes(y: np.ndarray) -> np.ndarray:
    """Bin returns into 5 classes per the CLASS_THRESHOLDS."""
    return np.searchsorted(CLASS_THRESHOLDS, y, side="right").astype(np.int64)


@dataclass
class TrainConfigV3:
    epochs: int = 60
    batch_size: int = 64
    lr: float = 5e-4
    weight_decay: float = 1e-5
    grad_clip: float = 1.0
    early_stopping_patience: int = 8
    seed: int = 42
    cls_alpha: float = 0.3           # weight on classification loss (after normalisation)
    use_balanced_sampler: bool = True
    mse_normalizer: float | None = None
    rcap_normalizer: float | None = None  # mean |r_cap| over training set
    cls_weights: torch.Tensor | None = None  # per-horizon (H, n_classes) inverse-freq weights


@dataclass
class TrainHistoryV3:
    train_loss: list[float] = field(default_factory=list)
    train_reg_loss: list[float] = field(default_factory=list)
    train_cls_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    val_reg_loss: list[float] = field(default_factory=list)
    val_cls_loss: list[float] = field(default_factory=list)
    epoch_times: list[float] = field(default_factory=list)
    best_epoch: int = -1
    best_val_loss: float = float("inf")
    total_seconds: float = 0.0


def _build_balanced_sampler(y_train: np.ndarray) -> WeightedRandomSampler:
    sign = np.sign(y_train[:, 0])
    sign[sign == 0] = 1
    classes, counts = np.unique(sign, return_counts=True)
    inv = {c: 1.0 / f for c, f in zip(classes.tolist(), counts.tolist())}
    weights = np.array([inv[s] for s in sign.tolist()], dtype=np.float64)
    return WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)


def _make_loaders(X, H, M, y, X_val, H_val, M_val, y_val, cfg: TrainConfigV3):
    train_ds = TensorDataset(
        torch.from_numpy(X), torch.from_numpy(H), torch.from_numpy(M),
        torch.from_numpy(y),
    )
    val_ds = TensorDataset(
        torch.from_numpy(X_val), torch.from_numpy(H_val), torch.from_numpy(M_val),
        torch.from_numpy(y_val),
    )
    if cfg.use_balanced_sampler:
        sampler = _build_balanced_sampler(y)
        train_dl = DataLoader(train_ds, batch_size=cfg.batch_size, sampler=sampler, drop_last=False)
    else:
        train_dl = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=False)
    val_dl = DataLoader(val_ds, batch_size=cfg.batch_size * 4, shuffle=False)
    return train_dl, val_dl


def _epoch(model, dl, device, cfg: TrainConfigV3, optimizer=None):
    is_train = optimizer is not None
    model.train(is_train)
    total = total_reg = total_cls = 0.0
    n = 0
    norm_mse = cfg.mse_normalizer or 1.0
    norm_w = cfg.rcap_normalizer or 1.0
    cls_weights = cfg.cls_weights.to(device) if cfg.cls_weights is not None else None
    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for xb, hb, mb, yb in dl:
            xb = xb.to(device); hb = hb.to(device); mb = mb.to(device); yb = yb.to(device)
            out = model(xb, hb, mb)

            reg_loss = F.mse_loss(out["reg"], yb) / norm_mse

            # Classification loss: per-horizon, per-sample CE * |r_cap| / mean(|r_cap|)
            r_cap = torch.clamp(yb, -0.5, 0.5).abs()  # (B, H)
            cls_target = torch.zeros_like(yb, dtype=torch.long)
            for i, t in enumerate(CLASS_THRESHOLDS):
                cls_target = cls_target + (yb >= t).long()
            cls_logits = out["cls_logits"]  # (B, H, n_classes)
            B, Hh, C = cls_logits.shape
            # Per-horizon CE with optional class weights
            ce_per = torch.zeros(B, Hh, device=device)
            for h in range(Hh):
                w_h = cls_weights[h] if cls_weights is not None else None
                ce_per[:, h] = F.cross_entropy(
                    cls_logits[:, h, :], cls_target[:, h],
                    weight=w_h, reduction="none",
                )
            cls_loss = (ce_per * (r_cap / norm_w)).mean()

            loss = reg_loss + cfg.cls_alpha * cls_loss
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                if cfg.grad_clip:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                optimizer.step()
            bs = xb.size(0)
            total += loss.item() * bs
            total_reg += reg_loss.item() * bs
            total_cls += cls_loss.item() * bs
            n += bs
    return total / max(n, 1), total_reg / max(n, 1), total_cls / max(n, 1)


def train_model_v3(
    model: nn.Module,
    X_train, H_train, M_train, y_train,
    X_val, H_val, M_val, y_val,
    cfg: TrainConfigV3,
    save_path: Path,
    device: str | None = None,
    on_epoch_end=None,
) -> TrainHistoryV3:
    set_seed(cfg.seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    if cfg.mse_normalizer is None:
        cfg.mse_normalizer = float(np.var(y_train, ddof=0).mean())
    if cfg.rcap_normalizer is None:
        cfg.rcap_normalizer = float(np.clip(np.abs(y_train), 0, 0.5).mean()) + 1e-6
    if cfg.cls_weights is None:
        # Per-horizon inverse-frequency class weights (sqrt to soften)
        n_h = y_train.shape[1]
        weights = np.zeros((n_h, len(CLASS_THRESHOLDS) + 1), dtype=np.float32)
        for h in range(n_h):
            cls = returns_to_classes(y_train[:, h])
            counts = np.bincount(cls, minlength=len(CLASS_THRESHOLDS) + 1) + 1
            inv = 1.0 / counts
            inv = np.sqrt(inv)
            inv = inv / inv.mean()  # normalise so mean weight = 1
            weights[h] = inv
        cfg.cls_weights = torch.tensor(weights, dtype=torch.float32)

    train_dl, val_dl = _make_loaders(X_train, H_train, M_train, y_train,
                                     X_val,   H_val,   M_val,   y_val, cfg)
    optim = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    hist = TrainHistoryV3()
    best_state = None
    epochs_since_improvement = 0
    t_start = time.time()
    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()
        tr_loss, tr_reg, tr_cls = _epoch(model, train_dl, device, cfg, optimizer=optim)
        va_loss, va_reg, va_cls = _epoch(model, val_dl, device, cfg)
        dt = time.time() - t0

        hist.train_loss.append(tr_loss); hist.train_reg_loss.append(tr_reg); hist.train_cls_loss.append(tr_cls)
        hist.val_loss.append(va_loss);   hist.val_reg_loss.append(va_reg);   hist.val_cls_loss.append(va_cls)
        hist.epoch_times.append(dt)

        improved = va_loss < hist.best_val_loss
        if improved:
            hist.best_val_loss = va_loss
            hist.best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1

        if on_epoch_end is not None:
            on_epoch_end(epoch, tr_loss, va_loss, dt, improved)

        if epochs_since_improvement >= cfg.early_stopping_patience:
            break

    hist.total_seconds = time.time() - t_start
    if best_state is not None:
        model.load_state_dict(best_state)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "state_dict": best_state,
            "best_val_loss": hist.best_val_loss,
            "best_epoch": hist.best_epoch,
            "epochs_run": len(hist.train_loss),
            "config": cfg.__dict__,
        }, save_path)
    return hist


@torch.no_grad()
def predict_v3(model, X, H, M, batch_size: int = 128, device: str | None = None) -> dict:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    reg_out, cls_out = [], []
    for b in range(0, len(X), batch_size):
        xb = torch.from_numpy(X[b: b + batch_size]).to(device)
        hb = torch.from_numpy(H[b: b + batch_size]).to(device)
        mb = torch.from_numpy(M[b: b + batch_size]).to(device)
        out = model(xb, hb, mb)
        reg_out.append(out["reg"].cpu().numpy())
        cls_out.append(F.softmax(out["cls_logits"], dim=-1).cpu().numpy())
    return {
        "reg": np.concatenate(reg_out, axis=0),
        "cls_prob": np.concatenate(cls_out, axis=0),  # (N, n_horizons, n_classes)
    }
