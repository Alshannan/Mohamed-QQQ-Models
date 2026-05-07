"""Mohamed's model: 14-feature fused transformer with optional LSTM branch.

Per the proposal:
- Linear projection (14 -> d_model) + LayerNorm + Dropout
- 20-day input sequence
- TransformerEncoder (2 layers, 4 heads, d_model=128, ffn=256)
- Optional LSTM branch on the 6 price features only, concatenated to the
  transformer output before the prediction heads
- Three separate heads -> t+1, t+3, t+5 returns
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


class MohamedModel(nn.Module):
    def __init__(
        self,
        n_features: int = 14,
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

        self.heads = nn.ModuleList(
            [nn.Sequential(nn.Linear(head_in, 64), nn.ReLU(), nn.Linear(64, 1))
             for _ in range(n_horizons)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F=14). First 6 columns are the price features.
        h = self.proj_drop(self.proj_norm(self.proj(x)))
        h = self.pos_enc(h)
        h = self.encoder(h)
        last = h[:, -1, :]

        if self.use_lstm_branch:
            price = x[:, :, : self.n_price_features]
            _, (hN, _) = self.lstm(price)
            last = torch.cat([last, hN[-1]], dim=-1)

        outs = [head(last) for head in self.heads]
        return torch.cat(outs, dim=-1)
