"""Training loop for MohamedModelV2.

Differences from src/train.py:
  - Combined loss: MSE on regression head + alpha * BCE on direction head.
  - Balanced WeightedRandomSampler on sign(y_t1) to fight the up-bias.
  - Returns predictions as a dict (reg + sign_prob) so downstream eval can use
    either head independently.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from .train import set_seed


@dataclass
class TrainConfigV2:
    epochs: int = 60
    batch_size: int = 64
    lr: float = 5e-4
    weight_decay: float = 1e-5
    grad_clip: float = 1.0
    early_stopping_patience: int = 8
    seed: int = 42
    bce_alpha: float = 0.5             # weight on direction loss (after MSE scaling)
    use_balanced_sampler: bool = True  # WeightedRandomSampler on sign(y_t1)
    # Raw MSE on returns is O(1e-4); raw BCE is O(0.5). To make the two
    # losses comparable we divide MSE by the per-horizon variance of y_train
    # so MSE starts near 1.0 - then bce_alpha is on a sane scale.
    mse_normalizer: float | None = None


@dataclass
class TrainHistoryV2:
    train_loss: list[float] = field(default_factory=list)
    train_reg_loss: list[float] = field(default_factory=list)
    train_bce_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    val_reg_loss: list[float] = field(default_factory=list)
    val_bce_loss: list[float] = field(default_factory=list)
    epoch_times: list[float] = field(default_factory=list)
    best_epoch: int = -1
    best_val_loss: float = float("inf")
    total_seconds: float = 0.0


def _build_balanced_sampler(y_train: np.ndarray) -> WeightedRandomSampler:
    """Inverse-frequency weights on sign(y_t1) to balance up vs down windows."""
    sign = np.sign(y_train[:, 0])
    sign[sign == 0] = 1
    classes, counts = np.unique(sign, return_counts=True)
    freq = dict(zip(classes.tolist(), counts.tolist()))
    inv = {c: 1.0 / f for c, f in freq.items()}
    weights = np.array([inv[s] for s in sign.tolist()], dtype=np.float64)
    return WeightedRandomSampler(
        weights=weights, num_samples=len(weights), replacement=True
    )


def _make_loaders(X_train, y_train, X_val, y_val, cfg: TrainConfigV2):
    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
    if cfg.use_balanced_sampler:
        sampler = _build_balanced_sampler(y_train)
        train_dl = DataLoader(train_ds, batch_size=cfg.batch_size, sampler=sampler, drop_last=False)
    else:
        train_dl = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=False)
    val_dl = DataLoader(val_ds, batch_size=cfg.batch_size * 4, shuffle=False)
    return train_dl, val_dl


def _epoch(model, dl, device, cfg: TrainConfigV2, optimizer=None, pos_weight=None):
    is_train = optimizer is not None
    model.train(is_train)
    mse_fn = nn.MSELoss()
    if pos_weight is not None:
        bce_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    else:
        bce_fn = nn.BCEWithLogitsLoss()
    total = total_reg = total_bce = 0.0
    n = 0
    norm = cfg.mse_normalizer if cfg.mse_normalizer is not None else 1.0
    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for xb, yb in dl:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            out = model(xb)
            reg_loss = mse_fn(out["reg"], yb) / norm
            sign_target = (yb > 0).float()
            bce_loss = bce_fn(out["sign_logits"], sign_target)
            loss = reg_loss + cfg.bce_alpha * bce_loss
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                if cfg.grad_clip:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                optimizer.step()
            bs = xb.size(0)
            total += loss.item() * bs
            total_reg += reg_loss.item() * bs
            total_bce += bce_loss.item() * bs
            n += bs
    return total / max(n, 1), total_reg / max(n, 1), total_bce / max(n, 1)


def train_model_v2(
    model: nn.Module,
    X_train, y_train, X_val, y_val,
    cfg: TrainConfigV2,
    save_path: Path,
    device: str | None = None,
    on_epoch_end=None,
) -> TrainHistoryV2:
    set_seed(cfg.seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    if cfg.mse_normalizer is None:
        # Variance of training returns averaged across horizons. Stable proxy
        # for the natural scale of the regression loss at convergence.
        cfg.mse_normalizer = float(np.var(y_train, ddof=0).mean())

    # NOTE: per-horizon BCE pos_weight was tested and made things worse -
    # the test period's up-rate (~55%) is close to train's, so rebalancing
    # the loss pushed the model to over-predict down. The WeightedRandomSampler
    # on sign(y_t1) is enough; we leave pos_weight=None.
    pos_weight = None

    train_dl, val_dl = _make_loaders(X_train, y_train, X_val, y_val, cfg)
    optim = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    hist = TrainHistoryV2()
    best_state = None
    epochs_since_improvement = 0

    t_start = time.time()
    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()
        tr_loss, tr_reg, tr_bce = _epoch(model, train_dl, device, cfg, optimizer=optim, pos_weight=pos_weight)
        va_loss, va_reg, va_bce = _epoch(model, val_dl, device, cfg, pos_weight=pos_weight)
        dt = time.time() - t0

        hist.train_loss.append(tr_loss); hist.train_reg_loss.append(tr_reg); hist.train_bce_loss.append(tr_bce)
        hist.val_loss.append(va_loss);   hist.val_reg_loss.append(va_reg);   hist.val_bce_loss.append(va_bce)
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
def predict_v2(model: nn.Module, X: np.ndarray, batch_size: int = 256, device: str | None = None) -> dict:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    reg, sig = [], []
    for b in range(0, len(X), batch_size):
        xb = torch.from_numpy(X[b : b + batch_size]).to(device)
        out = model(xb)
        reg.append(out["reg"].cpu().numpy())
        sig.append(torch.sigmoid(out["sign_logits"]).cpu().numpy())
    return {
        "reg": np.concatenate(reg, axis=0),
        "sign_prob": np.concatenate(sig, axis=0),
    }
