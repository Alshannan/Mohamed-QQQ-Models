"""MohamedModelV2: Mohamed's encoder + LSTM trunk with TWO output heads per horizon.

Tier-1 changes vs v1:
  - 15-feature input (adds confidence_weighted_net_sentiment).
  - Each horizon now has TWO heads:
        regression head     -> predicted return at t+k          (MSE loss)
        direction head      -> P(return at t+k > 0)             (BCE loss)
  - Combined loss: total = MSE + alpha * BCE
  - The direction head can't shrink to zero (BCE penalises 0.5 outputs),
    fixing the t+1 collapse the v1 model exhibited.

Forward returns a dict {'reg': (B, K), 'sign_logits': (B, K)} so callers can
choose which head drives a downstream decision (strategy uses sign_logits;
metrics use both).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 64):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class MohamedModelV2(nn.Module):
    def __init__(
        self,
        n_features: int = 15,
        n_price_features: int = 6,
        seq_len: int = 20,
        d_model: int = 128,
        n_heads: int = 4,
        ffn_dim: int = 256,
        n_layers: int = 2,
        dropout: float = 0.1,
        lstm_hidden: int = 64,
        use_lstm_branch: bool = True,
        n_horizons: int = 3,
    ):
        super().__init__()
        self.n_price_features = n_price_features
        self.use_lstm_branch = use_lstm_branch
        self.n_horizons = n_horizons

        self.proj = nn.Linear(n_features, d_model)
        self.proj_norm = nn.LayerNorm(d_model)
        self.proj_drop = nn.Dropout(dropout)

        self.pos_enc = SinusoidalPositionalEncoding(d_model, max_len=seq_len + 8)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        head_in = d_model
        if use_lstm_branch:
            self.lstm = nn.LSTM(
                input_size=n_price_features,
                hidden_size=lstm_hidden,
                num_layers=1,
                batch_first=True,
            )
            head_in += lstm_hidden

        # Per-horizon REGRESSION heads (predict return)
        self.reg_heads = nn.ModuleList([
            nn.Sequential(nn.Linear(head_in, 64), nn.ReLU(), nn.Linear(64, 1))
            for _ in range(n_horizons)
        ])
        # Per-horizon DIRECTION heads (predict logit of P(ret > 0))
        self.dir_heads = nn.ModuleList([
            nn.Sequential(nn.Linear(head_in, 64), nn.ReLU(), nn.Linear(64, 1))
            for _ in range(n_horizons)
        ])

    def forward(self, x: torch.Tensor) -> dict:
        # x: (B, T, F=15). First 6 columns are price features.
        h = self.proj_drop(self.proj_norm(self.proj(x)))
        h = self.pos_enc(h)
        h = self.encoder(h)
        last = h[:, -1, :]

        if self.use_lstm_branch:
            price = x[:, :, : self.n_price_features]
            _, (hN, _) = self.lstm(price)
            last = torch.cat([last, hN[-1]], dim=-1)

        reg = torch.cat([head(last) for head in self.reg_heads], dim=-1)
        sign_logits = torch.cat([head(last) for head in self.dir_heads], dim=-1)
        return {"reg": reg, "sign_logits": sign_logits}
