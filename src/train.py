"""Training loop helpers used by the training notebooks.

Notebooks own all plotting; this module owns the loop, optimizer, and
checkpointing logic so both runs share an identical training procedure.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class TrainConfig:
    epochs: int = 60
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-5
    grad_clip: float = 1.0
    early_stopping_patience: int = 8
    seed: int = 42


@dataclass
class TrainHistory:
    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    epoch_times: list[float] = field(default_factory=list)
    best_epoch: int = -1
    best_val_loss: float = float("inf")
    total_seconds: float = 0.0


def set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_loaders(X_train, y_train, X_val, y_val, batch_size: int):
    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_dl = DataLoader(val_ds, batch_size=batch_size * 4, shuffle=False)
    return train_dl, val_dl


def _epoch(model, dl, device, optimizer=None, grad_clip: float | None = None):
    is_train = optimizer is not None
    model.train(is_train)
    loss_fn = nn.MSELoss()
    total = 0.0
    n = 0
    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for xb, yb in dl:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            preds = model(xb)
            loss = loss_fn(preds, yb)
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                if grad_clip:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
            total += loss.item() * xb.size(0)
            n += xb.size(0)
    return total / max(n, 1)


def train_model(
    model: nn.Module,
    X_train, y_train, X_val, y_val,
    cfg: TrainConfig,
    save_path: Path,
    device: str | None = None,
    on_epoch_end=None,
) -> TrainHistory:
    set_seed(cfg.seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    train_dl, val_dl = make_loaders(X_train, y_train, X_val, y_val, cfg.batch_size)
    optim = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    history = TrainHistory()
    best_state = None
    epochs_since_improvement = 0

    t_start = time.time()
    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()
        train_loss = _epoch(model, train_dl, device, optimizer=optim, grad_clip=cfg.grad_clip)
        val_loss = _epoch(model, val_dl, device)
        dt = time.time() - t0

        history.train_loss.append(train_loss)
        history.val_loss.append(val_loss)
        history.epoch_times.append(dt)

        improved = val_loss < history.best_val_loss
        if improved:
            history.best_val_loss = val_loss
            history.best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1

        if on_epoch_end is not None:
            on_epoch_end(epoch, train_loss, val_loss, dt, improved)

        if epochs_since_improvement >= cfg.early_stopping_patience:
            break

    history.total_seconds = time.time() - t_start
    if best_state is not None:
        model.load_state_dict(best_state)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "state_dict": best_state,
            "best_val_loss": history.best_val_loss,
            "best_epoch": history.best_epoch,
            "epochs_run": len(history.train_loss),
            "config": cfg.__dict__,
        }, save_path)
    return history


@torch.no_grad()
def predict(model: nn.Module, X: np.ndarray, batch_size: int = 256, device: str | None = None) -> np.ndarray:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    out = []
    for b in range(0, len(X), batch_size):
        xb = torch.from_numpy(X[b : b + batch_size]).to(device)
        out.append(model(xb).cpu().numpy())
    return np.concatenate(out, axis=0)
