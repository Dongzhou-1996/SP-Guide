from __future__ import annotations

import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F


def pairwise_normalized_distance(query_coords: Tensor, key_coords: Tensor) -> Tensor:
    """Return pairwise distances between normalized 2D patch centers.

    Parameters
    ----------
    query_coords:
        Tensor of shape ``[B, Q, 2]`` in a shared global coordinate frame.
    key_coords:
        Tensor of shape ``[B, K, 2]`` in the same coordinate frame.
    """
    if query_coords.ndim != 3 or key_coords.ndim != 3:
        raise ValueError("coords must have shape [B, N, 2]")
    return torch.cdist(query_coords.clamp(0.0, 1.0), key_coords.clamp(0.0, 1.0), p=2)


class UnifiedSpatialConstraintAttention(nn.Module):
    """Attention with a soft spatial-prior reweighting operator."""

    def __init__(self, hidden_size: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

        self.alpha_logit = nn.Parameter(torch.tensor(-2.0))
        self.distance_scale = nn.Parameter(torch.tensor(0.35))

    def forward(self, query_tokens: Tensor, key_tokens: Tensor, distances: Tensor) -> Tensor:
        batch_size, num_queries, _ = query_tokens.shape
        num_keys = key_tokens.size(1)

        q = self.q_proj(query_tokens).view(batch_size, num_queries, self.num_heads, self.head_dim)
        k = self.k_proj(key_tokens).view(batch_size, num_keys, self.num_heads, self.head_dim)
        v = self.v_proj(key_tokens).view(batch_size, num_keys, self.num_heads, self.head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attention = scores.softmax(dim=-1)

        spatial_weight = torch.exp(-distances / F.softplus(self.distance_scale).clamp_min(1e-4))
        spatial_weight = 0.5 * (spatial_weight + spatial_weight.mean(dim=-1, keepdim=True))
        alpha = torch.sigmoid(self.alpha_logit)

        attention = attention * (1.0 + alpha * (spatial_weight.unsqueeze(1) - 1.0))
        attention = attention.clamp_min(1e-6)
        attention = attention / attention.sum(dim=-1, keepdim=True)
        attention = self.dropout(attention)

        output = torch.matmul(attention, v)
        output = output.transpose(1, 2).reshape(batch_size, num_queries, self.hidden_size)
        return self.out_proj(output)


class SpatialPriorCrossModalEncoder(nn.Module):
    """Refine RGB tokens by attending to map tokens under shared spatial priors."""

    def __init__(self, hidden_size: int, num_heads: int, num_layers: int = 1, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            _SpatialPriorBlock(hidden_size, num_heads, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, rgb_tokens: Tensor, map_tokens: Tensor, rgb_coords: Tensor, map_coords: Tensor) -> Tensor:
        distances = pairwise_normalized_distance(rgb_coords, map_coords)
        for layer in self.layers:
            rgb_tokens = layer(rgb_tokens, map_tokens, distances)
        return rgb_tokens


class _SpatialPriorBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, dropout: float):
        super().__init__()
        self.query_norm = nn.LayerNorm(hidden_size)
        self.key_norm = nn.LayerNorm(hidden_size)
        self.attn = UnifiedSpatialConstraintAttention(hidden_size, num_heads, dropout)
        self.ffn_norm = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, hidden_size),
            nn.Dropout(dropout),
        )

    def forward(self, query_tokens: Tensor, key_tokens: Tensor, distances: Tensor) -> Tensor:
        query_tokens = query_tokens + self.attn(
            self.query_norm(query_tokens),
            self.key_norm(key_tokens),
            distances,
        )
        return query_tokens + self.ffn(self.ffn_norm(query_tokens))

