"""Baseline price-only transformer (matches Bao's milestone baseline).

Input window: (B, 30, 6) on Open/High/Low/Close/Volume/MACD.
One transformer block (2 heads, d_model=128) -> last timestep -> 3 returns.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class BaselineTransformer(nn.Module):
    def __init__(
        self,
        n_features: int = 6,
        seq_len: int = 30,
        d_model: int = 128,
        n_heads: int = 2,
        ffn_dim: int = 128,
        dropout: float = 0.1,
        n_horizons: int = 3,
    ):
        super().__init__()
        self.proj = nn.Linear(n_features, d_model)
        self.pos_emb = nn.Embedding(seq_len, d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.ReLU(),
            nn.Linear(ffn_dim, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, n_horizons),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F)
        b, t, _ = x.shape
        h = self.proj(x)
        positions = torch.arange(t, device=x.device).unsqueeze(0).expand(b, t)
        h = h + self.pos_emb(positions)
        # Self-attention block
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        h = self.norm1(h + attn_out)
        ffn_out = self.ffn(h)
        h = self.norm2(h + ffn_out)
        last = h[:, -1, :]
        return self.head(last)
