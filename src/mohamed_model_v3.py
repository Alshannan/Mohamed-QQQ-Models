"""MohamedModelV3 - Tier-A v3 architecture.

Three structural changes vs v2:

  (1) MARKET-NEWS ATTENTION POOL  (MANA-Net pattern, arXiv 2409.05698):
      For each of the T days in the window, take the (MAX_HEADLINES, 4)
      per-headline FinBERT features and pool them with attention. The
      query comes from the day's price embedding so the pool weighting
      is conditioned on market state. Difference-enlargement softmax
      sharpens weights so a few important headlines dominate. The
      pooled (D_attn,) vector is concatenated to the day's base feature
      vector before the encoder trunk.

  (2) ENRICHED FEATURE BLOCK  (29 base features instead of 15):
      Adds 9 technical indicators and 5 macro features (lagged 1 day) on
      top of the v2 price + sentiment block. See src/macro_features.py.

  (3) FIVE-CLASS DIRECTION HEAD per horizon  (return-weighted CE,
      Novel Loss Function paper, arXiv 2502.17493):
      Each horizon outputs a 5-class softmax over
      [Strong Sell, Sell, Hold, Buy, Strong Buy]. The classification
      loss is cross-entropy weighted by |capped return|, so the
      model is rewarded most for getting the *big-move* days right -
      the days that drive realised P&L.

  The regression head is kept (predicting return value) for $-space
  conversion and persistence comparisons; the v2 sigmoid binary head
  is replaced by the 5-class head.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

NEG_INF = -1e9


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 64):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class MarketNewsAttentionPool(nn.Module):
    """Per-day attention pool over the day's headlines (MANA-Net pattern).

    Inputs:
      base_day:   (B, T, F_base)  base feature vector per day (price-block side)
      headlines:  (B, T, H, S)    per-headline sentiment vectors (MAX_HEADLINES, sentiment_dim)
      mask:       (B, T, H)       True = padding to ignore
    Output:
      pooled:     (B, T, D_attn)  attention-pooled headline vector per day
    """
    def __init__(
        self,
        base_dim: int,
        sentiment_dim: int = 4,
        d_model: int = 32,
        n_heads: int = 2,
        diff_enlarge: float = 1.5,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.diff_enlarge = diff_enlarge
        assert d_model % n_heads == 0
        self.head_dim = d_model // n_heads

        self.q_proj = nn.Linear(base_dim, d_model, bias=False)
        self.k_proj = nn.Linear(sentiment_dim, d_model, bias=False)
        self.v_proj = nn.Linear(sentiment_dim, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, base_day, headlines, mask):
        B, T, H, S = headlines.shape
        # Q from per-day base features: (B, T, D)
        Q = self.q_proj(base_day)
        # K, V from per-headline features: (B, T, H, D)
        K = self.k_proj(headlines)
        V = self.v_proj(headlines)

        # Reshape into multi-head form
        # Q: (B*T, n_heads, head_dim)  one query per day
        Q = Q.view(B * T, self.n_heads, self.head_dim)
        K = K.view(B * T, H, self.n_heads, self.head_dim).transpose(1, 2)  # (B*T, n_heads, H, head_dim)
        V = V.view(B * T, H, self.n_heads, self.head_dim).transpose(1, 2)
        flat_mask = mask.view(B * T, H)  # True = padding

        # Scaled dot-product per day: (B*T, n_heads, H)
        scores = torch.einsum("bnd,bnhd->bnh", Q, K) / math.sqrt(self.head_dim)
        scores = scores * self.diff_enlarge
        scores = scores.masked_fill(flat_mask.unsqueeze(1), NEG_INF)
        # If a day has ALL headlines masked (no news at all that day), softmax
        # would explode; handle by checking and producing zeros.
        all_masked = flat_mask.all(dim=-1, keepdim=True)  # (B*T, 1)
        attn = torch.softmax(scores, dim=-1)
        attn = attn.masked_fill(all_masked.unsqueeze(1), 0.0)

        # Weighted sum: (B*T, n_heads, head_dim)
        pooled = torch.einsum("bnh,bnhd->bnd", attn, V)
        pooled = pooled.contiguous().view(B * T, self.d_model)
        pooled = self.out_proj(pooled)
        return pooled.view(B, T, self.d_model)


class MohamedModelV3(nn.Module):
    def __init__(
        self,
        n_features: int = 29,
        n_price_features: int = 6,
        sentiment_dim: int = 4,
        max_headlines: int = 16,
        seq_len: int = 20,
        d_model: int = 128,
        n_heads: int = 4,
        ffn_dim: int = 256,
        n_layers: int = 2,
        dropout: float = 0.1,
        lstm_hidden: int = 64,
        attn_pool_dim: int = 32,
        n_horizons: int = 3,
        n_classes: int = 5,
    ):
        super().__init__()
        self.n_price_features = n_price_features
        self.n_horizons = n_horizons
        self.n_classes = n_classes

        # Per-day attention pool over headlines
        self.attn_pool = MarketNewsAttentionPool(
            base_dim=n_features,
            sentiment_dim=sentiment_dim,
            d_model=attn_pool_dim,
            n_heads=2,
        )

        # Encoder trunk receives base_features ++ pooled_attn
        proj_in_dim = n_features + attn_pool_dim
        self.proj = nn.Linear(proj_in_dim, d_model)
        self.proj_norm = nn.LayerNorm(d_model)
        self.proj_drop = nn.Dropout(dropout)

        self.pos_enc = SinusoidalPositionalEncoding(d_model, max_len=seq_len + 8)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=ffn_dim, dropout=dropout,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.lstm = nn.LSTM(
            input_size=n_price_features, hidden_size=lstm_hidden,
            num_layers=1, batch_first=True,
        )
        head_in = d_model + lstm_hidden

        # REGRESSION heads (continuous return)
        self.reg_heads = nn.ModuleList([
            nn.Sequential(nn.Linear(head_in, 64), nn.ReLU(), nn.Linear(64, 1))
            for _ in range(n_horizons)
        ])
        # 5-CLASS heads (Strong Sell .. Strong Buy)
        self.cls_heads = nn.ModuleList([
            nn.Sequential(nn.Linear(head_in, 64), nn.ReLU(), nn.Linear(64, n_classes))
            for _ in range(n_horizons)
        ])

    def forward(self, x: torch.Tensor, headlines: torch.Tensor, mask: torch.Tensor) -> dict:
        # x: (B, T, F)
        # headlines: (B, T, H, S)
        # mask: (B, T, H)  True = padding
        pooled = self.attn_pool(x, headlines, mask)            # (B, T, D_attn)
        h = torch.cat([x, pooled], dim=-1)                      # (B, T, F + D_attn)

        h = self.proj_drop(self.proj_norm(self.proj(h)))
        h = self.pos_enc(h)
        h = self.encoder(h)
        last = h[:, -1, :]

        price = x[:, :, : self.n_price_features]
        _, (hN, _) = self.lstm(price)
        last = torch.cat([last, hN[-1]], dim=-1)

        reg = torch.cat([head(last) for head in self.reg_heads], dim=-1)
        cls_logits = torch.stack([head(last) for head in self.cls_heads], dim=1)  # (B, n_horizons, n_classes)
        return {"reg": reg, "cls_logits": cls_logits}
