from __future__ import annotations

import torch
from torch import Tensor, nn

from .spce import SpatialPriorCrossModalEncoder


class InstructionQueryDecoder(nn.Module):
    """Decode a navigation summary using language tokens as queries."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        layer = nn.TransformerDecoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(
        self,
        instruction_tokens: Tensor,
        visual_memory: Tensor,
        instruction_padding_mask: Tensor | None = None,
    ) -> Tensor:
        decoded = self.decoder(
            tgt=instruction_tokens,
            memory=visual_memory,
            tgt_key_padding_mask=instruction_padding_mask,
        )
        if instruction_padding_mask is None:
            summary = decoded.mean(dim=1)
        else:
            valid = (~instruction_padding_mask).to(decoded.dtype).unsqueeze(-1)
            summary = (decoded * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)
        return self.norm(summary)


class SPGuide(nn.Module):
    """Minimal SP-Guide model over precomputed modality tokens.

    The model expects external encoders to provide token sequences for language,
    map, RGB, and depth. This keeps the core method independent of any specific
    pretrained visual or language backbone.
    """

    def __init__(
        self,
        hidden_size: int = 256,
        num_heads: int = 8,
        spce_layers: int = 1,
        visual_encoder_layers: int = 2,
        decoder_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.spce = SpatialPriorCrossModalEncoder(hidden_size, num_heads, spce_layers, dropout)
        self.depth_pool = nn.Sequential(nn.LayerNorm(hidden_size), nn.Linear(hidden_size, hidden_size), nn.GELU())
        self.rgb_pool = nn.Sequential(nn.LayerNorm(hidden_size), nn.Linear(hidden_size, hidden_size), nn.GELU())
        self.map_pool = nn.Sequential(nn.LayerNorm(hidden_size), nn.Linear(hidden_size, hidden_size), nn.GELU())
        self.modality_embed = nn.Embedding(3, hidden_size)
        self.visual_pos = nn.Parameter(torch.zeros(1, 3, hidden_size))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.visual_encoder = nn.TransformerEncoder(encoder_layer, num_layers=visual_encoder_layers)
        self.visual_norm = nn.LayerNorm(hidden_size)
        self.decoder = InstructionQueryDecoder(hidden_size, num_heads, decoder_layers, dropout)

        self.goal_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 2),
            nn.Sigmoid(),
        )
        self.progress_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid(),
        )

        nn.init.trunc_normal_(self.visual_pos, std=0.02)

    def forward(
        self,
        instruction_tokens: Tensor,
        map_tokens: Tensor,
        rgb_tokens: Tensor,
        depth_tokens: Tensor,
        map_coords: Tensor,
        rgb_coords: Tensor,
        depth_coords: Tensor | None = None,
        instruction_padding_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        del depth_coords
        rgb_tokens = self.spce(rgb_tokens, map_tokens, rgb_coords, map_coords)

        visual_tokens = torch.cat(
            (
                self.map_pool(map_tokens.mean(dim=1, keepdim=True)),
                self.rgb_pool(rgb_tokens.mean(dim=1, keepdim=True)),
                self.depth_pool(depth_tokens.mean(dim=1, keepdim=True)),
            ),
            dim=1,
        )
        visual_tokens = visual_tokens + self.visual_pos + self.modality_embed.weight.unsqueeze(0)
        memory = self.visual_norm(self.visual_encoder(visual_tokens))
        summary = self.decoder(instruction_tokens, memory, instruction_padding_mask)
        return self.goal_head(summary), self.progress_head(summary)

