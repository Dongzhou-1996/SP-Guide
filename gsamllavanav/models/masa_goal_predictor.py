from __future__ import annotations

import math
from collections.abc import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .ddppo.resenet_encoders import ResnetDepthEncoder, TorchVisionResNet50


class MapPatchEncoder(nn.Module):
    """CityNav 5-channel map encoder that keeps the final 15x15 patch grid."""

    def __init__(self, map_size: int):
        super().__init__()
        self.main = nn.Sequential(
            nn.MaxPool2d(2), nn.Conv2d(5, 32, 3, stride=1, padding=1), nn.ReLU(),
            nn.MaxPool2d(2), nn.Conv2d(32, 64, 3, stride=1, padding=1), nn.ReLU(),
            nn.MaxPool2d(2), nn.Conv2d(64, 128, 3, stride=1, padding=1), nn.ReLU(),
            nn.MaxPool2d(2), nn.Conv2d(128, 64, 3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(64, 32, 3, stride=1, padding=1), nn.ReLU(),
        )
        self.grid_size = map_size // 2**4
        self.out_channels = 32
        self.out_features = self.grid_size * self.grid_size * self.out_channels

    def forward(self, maps: Tensor) -> Tensor:
        patch_grid = self.main(maps)
        return patch_grid.flatten(2).transpose(1, 2)


class SpatiallyWeightedSelfAttention(nn.Module):
    """Self-attention with multiplicative spatial weighting.

    A = softmax(QK^T)
    D = spatial_weight(distance)
    A' = normalize(A * (1 + alpha * (D - 1)))
    out = A'V
    """

    def __init__(self, hidden_size: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.qkv = nn.Linear(hidden_size, hidden_size * 3)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.alpha_logit = nn.Parameter(torch.tensor(-2.0))
        self.distance_scale = nn.Parameter(torch.tensor(0.5))

    def forward(self, tokens: Tensor, pairwise_distances: Tensor) -> Tensor:
        batch_size, num_tokens, _ = tokens.shape
        qkv = self.qkv(tokens).view(batch_size, num_tokens, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attention = scores.softmax(dim=-1)
        spatial_weight = torch.exp(-pairwise_distances / F.softplus(self.distance_scale).clamp_min(1e-4))
        alpha = torch.sigmoid(self.alpha_logit)
        attention = attention * (1.0 + alpha * (spatial_weight.unsqueeze(0).unsqueeze(0) - 1.0))
        attention = attention.clamp_min(1e-6)
        attention = attention / attention.sum(dim=-1, keepdim=True)
        attention = self.dropout(attention)

        attended = torch.matmul(attention, v).transpose(1, 2).reshape(batch_size, num_tokens, self.hidden_size)
        return self.out_proj(attended)


class MasaMapBlock(nn.Module):

    def __init__(self, hidden_size: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size)
        self.attn = SpatiallyWeightedSelfAttention(hidden_size, num_heads, dropout)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, hidden_size),
            nn.Dropout(dropout),
        )

    def forward(self, tokens: Tensor, pairwise_distances: Tensor) -> Tensor:
        tokens = tokens + self.attn(self.norm1(tokens), pairwise_distances)
        tokens = tokens + self.ffn(self.norm2(tokens))
        return tokens


class MasaGoalPredictor(nn.Module):

    def __init__(
        self,
        map_size: int,
        hidden_size: int = 128,
        num_heads: int = 4,
        depth: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.map_encoder = MapPatchEncoder(map_size)
        self.rgb_encoder = TorchVisionResNet50().eval()
        self.depth_encoder = ResnetDepthEncoder().eval()

        self.map_proj = nn.Linear(self.map_encoder.out_channels, hidden_size)
        self.map_pos = nn.Parameter(torch.zeros(1, self.map_encoder.grid_size**2, hidden_size))
        self.masa_blocks = nn.ModuleList([
            MasaMapBlock(hidden_size, num_heads, dropout)
            for _ in range(depth)
        ])
        self.map_norm = nn.LayerNorm(hidden_size)
        self.map_pool = nn.Linear(hidden_size, 1)
        self.map_out = nn.Linear(hidden_size, self.map_encoder.out_features)

        coords = self._build_grid_coords(self.map_encoder.grid_size)
        self.register_buffer("pairwise_distances", torch.cdist(coords, coords, p=2), persistent=False)

        self.goal_prediction_head = nn.Sequential(
            nn.Linear(self.map_encoder.out_features, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, 2), nn.Sigmoid(),
        )
        self.progress_prediction_head = nn.Sequential(
            nn.Linear(self.map_encoder.out_features + self.rgb_encoder.out_features + self.depth_encoder.out_features, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )
        nn.init.trunc_normal_(self.map_pos, std=0.02)

    @staticmethod
    def _build_grid_coords(grid_size: int) -> Tensor:
        lin = torch.linspace(0.0, 1.0, grid_size)
        yy, xx = torch.meshgrid(lin, lin, indexing="ij")
        return torch.stack((xx, yy), dim=-1).reshape(-1, 2)

    def _encode_map(self, maps: Tensor) -> Tensor:
        map_tokens = self.map_proj(self.map_encoder(maps)) + self.map_pos
        for block in self.masa_blocks:
            map_tokens = block(map_tokens, self.pairwise_distances)
        map_tokens = self.map_norm(map_tokens)
        pool_logits = self.map_pool(map_tokens).squeeze(-1)
        pooled_map = torch.sum(map_tokens * pool_logits.softmax(dim=1).unsqueeze(-1), dim=1)
        return self.map_out(pooled_map)

    def forward(self, maps: Tensor, rgbs: Tensor, depths: Tensor, flip_depth=True):
        if flip_depth:
            depths = depths.flip(-2)

        map_features = self._encode_map(maps)
        rgb_features = self.rgb_encoder(rgbs)
        depth_features = self.depth_encoder(depths)

        pred_normalized_goal_xys = self.goal_prediction_head(map_features)
        pred_progress = self.progress_prediction_head(torch.cat((map_features, rgb_features, depth_features), dim=1))
        return pred_normalized_goal_xys, pred_progress

    def predict(
        self,
        to_world_xy: Callable[[tuple[float, float]], tuple[float, float]],
        maps: Tensor,
        rgb: Tensor,
        depth: Tensor,
        flip_depth=True,
    ):
        pred_normalized_goal_coords, pred_progress = self(maps, rgb, depth, flip_depth)
        pred_xy = to_world_xy(pred_normalized_goal_coords)
        return pred_xy, pred_progress
