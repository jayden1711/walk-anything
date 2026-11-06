"""Morphology-token transformer policy: one token per joint plus a global token. Same weights drive every body."""
from __future__ import annotations

import torch
from torch import nn

from .env import GLOBAL_DIM, JOINT_DIM
from .spec import J_MAX


class MorphPolicy(nn.Module):
    def __init__(self, d: int = 64, layers: int = 2, heads: int = 4, init_log_std: float = -0.7, joint_dim: int = JOINT_DIM):
        super().__init__()
        self.g_emb = nn.Linear(GLOBAL_DIM, d)
        self.j_emb = nn.Sequential(nn.Linear(joint_dim, d), nn.ReLU(), nn.Linear(d, d))
        layer = nn.TransformerEncoderLayer(d, heads, dim_feedforward=2 * d, dropout=0.0, batch_first=True, norm_first=True, activation="gelu")
        self.enc = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)
        self.pi = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, 1))
        self.v = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, 1))
        self.log_std = nn.Parameter(torch.full((1,), init_log_std))
        nn.init.normal_(self.pi[-1].weight, std=0.01)
        nn.init.zeros_(self.pi[-1].bias)

    def forward(self, g: torch.Tensor, j: torch.Tensor, mask: torch.Tensor):
        """g (B, GLOBAL_DIM), j (B, J_MAX, JOINT_DIM), mask (B, J_MAX) 1 = real joint. Returns mean (B, J_MAX), value (B,)."""
        tok = torch.cat([self.g_emb(g)[:, None], self.j_emb(j)], dim=1)
        pad = torch.cat([torch.zeros_like(mask[:, :1]), 1 - mask], dim=1).bool()
        h = self.norm(self.enc(tok, src_key_padding_mask=pad))
        return self.pi(h[:, 1:]).squeeze(-1) * mask, self.v(h[:, 0]).squeeze(-1)

    def dist_stats(self, mean, mask, action):
        """Masked diagonal Gaussian: log prob (B,), entropy (B,)."""
        std = self.log_std.exp()
        lp = (-0.5 * ((action - mean) / std) ** 2 - self.log_std - 0.9189385) * mask
        ent = (self.log_std + 1.4189385) * mask
        return lp.sum(-1), ent.sum(-1)
