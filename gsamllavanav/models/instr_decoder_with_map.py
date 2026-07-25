from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .ddppo.resenet_encoders import ResnetDepthEncoder, TorchVisionResNet50
from .goal_predictor import MapEncoder
from .masa_goal_predictor import MapPatchEncoder
from .seq2seq_with_map import InstructionEncoder


class InstructionQueryDecoderWithMap(nn.Module):
    """CityNav MGP encoders with encoder-decoder fusion.

    This variant keeps CityNav's original map/RGB/depth embedding modules:
    `MapEncoder`, `TorchVisionResNet50`, and `ResnetDepthEncoder`. The only
    changed part is the fusion stage: the three visual embeddings become visual
    tokens, and instruction embeddings query them through a Transformer decoder.
    """

    def __init__(
        self,
        map_size: int,
        hidden_size: int = 512,
        num_heads: int = 8,
        encoder_layers: int = 2,
        decoder_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_size = hidden_size

        self.instruction_encoder = InstructionEncoder(
            final_state_only=False,
            bidirectional=True,
            vocab_size=30522,
        )
        self.map_encoder = MapEncoder(map_size)
        self.rgb_encoder = TorchVisionResNet50().eval()
        self.depth_encoder = ResnetDepthEncoder().eval()

        self.map_proj = nn.Linear(self.map_encoder.out_features, hidden_size)
        self.rgb_proj = nn.Linear(self.rgb_encoder.out_features, hidden_size)
        self.depth_proj = nn.Linear(self.depth_encoder.out_features, hidden_size)
        self.instruction_proj = nn.Linear(self.instruction_encoder.output_size, hidden_size)

        self.visual_pos = nn.Parameter(torch.zeros(1, 3, hidden_size))
        self.modality_embed = nn.Embedding(3, hidden_size)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.visual_encoder = nn.TransformerEncoder(encoder_layer, num_layers=encoder_layers)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.instruction_decoder = nn.TransformerDecoder(decoder_layer, num_layers=decoder_layers)
        self.visual_norm = nn.LayerNorm(hidden_size)
        self.summary_norm = nn.LayerNorm(hidden_size)

        self.goal_prediction_head = nn.Sequential(
            nn.Linear(hidden_size, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, 2), nn.Sigmoid(),
        )
        self.progress_prediction_head = nn.Sequential(
            nn.Linear(hidden_size, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, 1), nn.Sigmoid(),
        )

        nn.init.trunc_normal_(self.visual_pos, std=0.02)
        self.train()

    @property
    def num_recurrent_layers(self) -> int:
        return 1

    def _masked_mean(self, tokens: Tensor, valid_mask: Tensor) -> Tensor:
        weights = valid_mask.to(tokens.dtype).unsqueeze(-1)
        return (tokens * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)

    def forward(
        self,
        tokenized_instruction_batch: Tensor,
        depth_batch: Tensor,
        rgb_batch: Tensor,
        map_batch: Tensor,
        rnn_states_batch: Tensor,
        masks: Tensor,
    ):
        depth_batch = depth_batch.flip(-2)

        instruction_tokens = self.instruction_encoder(tokenized_instruction_batch).transpose(1, 2)
        instruction_tokens = self.instruction_proj(instruction_tokens)
        instruction_padding_mask = tokenized_instruction_batch[:, :instruction_tokens.size(1)].eq(0)
        instruction_valid_mask = ~instruction_padding_mask

        map_token = self.map_proj(self.map_encoder(map_batch)).unsqueeze(1)
        rgb_token = self.rgb_proj(self.rgb_encoder(rgb_batch)).unsqueeze(1)
        depth_token = self.depth_proj(self.depth_encoder(depth_batch)).unsqueeze(1)
        visual_tokens = torch.cat((map_token, rgb_token, depth_token), dim=1)
        visual_tokens = visual_tokens + self.visual_pos + self.modality_embed.weight.unsqueeze(0)

        memory = self.visual_norm(self.visual_encoder(visual_tokens))
        decoded_instruction = self.instruction_decoder(
            tgt=instruction_tokens,
            memory=memory,
            tgt_key_padding_mask=instruction_padding_mask,
        )
        summary = self.summary_norm(self._masked_mean(decoded_instruction, instruction_valid_mask))

        pred_normalized_goal_xys = self.goal_prediction_head(summary)
        pred_progress = self.progress_prediction_head(summary)
        return pred_normalized_goal_xys, pred_progress, rnn_states_batch

    def get_initial_recurrent_hidden_states(self, batch_size: int, device: str):
        return torch.zeros(batch_size, self.num_recurrent_layers, 1, device=device)


class CPVTPositionEncodingGenerator(nn.Module):
    """CPVT-style Position Encoding Generator.

    PEG generates conditional positional encodings from local neighborhoods with
    a depth-wise 3x3 convolution and adds them back to the feature grid.
    """

    def __init__(self, channels: int, kernel_size: int = 3):
        super().__init__()
        self.proj = nn.Conv2d(
            channels,
            channels,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            groups=channels,
            bias=True,
        )

    def forward(self, feature_grid: Tensor) -> Tensor:
        return feature_grid + self.proj(feature_grid)


class InstructionQueryDecoderWithCPVTMap(nn.Module):
    """Ins-Dec with CPVT conditional position encoding before modality pooling.

    The instruction-query decoder and three-token visual encoder are unchanged
    from `InstructionQueryDecoderWithMap`. CPVT/PEG is only inserted inside each
    visual stream before map flattening or RGB/depth global pooling.
    """

    def __init__(
        self,
        map_size: int,
        hidden_size: int = 512,
        num_heads: int = 8,
        encoder_layers: int = 2,
        decoder_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_size = hidden_size

        self.instruction_encoder = InstructionEncoder(
            final_state_only=False,
            bidirectional=True,
            vocab_size=30522,
        )
        self.map_patch_encoder = MapPatchEncoder(map_size)
        self.rgb_encoder = TorchVisionResNet50().eval()
        self.depth_encoder = ResnetDepthEncoder().eval()

        rgb_grid_channels = self.rgb_encoder.fc[1].in_features
        depth_grid_channels = self.depth_encoder.visual_encoder.output_shape[0]
        self.map_peg = CPVTPositionEncodingGenerator(self.map_patch_encoder.out_channels)
        self.rgb_peg = CPVTPositionEncodingGenerator(rgb_grid_channels)
        self.depth_peg = CPVTPositionEncodingGenerator(depth_grid_channels)

        self.map_proj = nn.Linear(self.map_patch_encoder.out_features, hidden_size)
        self.rgb_proj = nn.Linear(self.rgb_encoder.out_features, hidden_size)
        self.depth_proj = nn.Linear(self.depth_encoder.out_features, hidden_size)
        self.instruction_proj = nn.Linear(self.instruction_encoder.output_size, hidden_size)

        self.visual_pos = nn.Parameter(torch.zeros(1, 3, hidden_size))
        self.modality_embed = nn.Embedding(3, hidden_size)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.visual_encoder = nn.TransformerEncoder(encoder_layer, num_layers=encoder_layers)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.instruction_decoder = nn.TransformerDecoder(decoder_layer, num_layers=decoder_layers)
        self.visual_norm = nn.LayerNorm(hidden_size)
        self.summary_norm = nn.LayerNorm(hidden_size)

        self.goal_prediction_head = nn.Sequential(
            nn.Linear(hidden_size, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, 2), nn.Sigmoid(),
        )
        self.progress_prediction_head = nn.Sequential(
            nn.Linear(hidden_size, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, 1), nn.Sigmoid(),
        )

        nn.init.trunc_normal_(self.visual_pos, std=0.02)
        self.train()

    @property
    def num_recurrent_layers(self) -> int:
        return 1

    def _masked_mean(self, tokens: Tensor, valid_mask: Tensor) -> Tensor:
        weights = valid_mask.to(tokens.dtype).unsqueeze(-1)
        return (tokens * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)

    def _encode_map_with_peg(self, map_batch: Tensor) -> Tensor:
        map_grid = self.map_patch_encoder.main(map_batch)
        map_grid = self.map_peg(map_grid)
        return map_grid.flatten(1)

    def _encode_rgb_with_peg(self, rgb_batch: Tensor) -> Tensor:
        rgb = rgb_batch.contiguous() / 255.0
        rgb = self.rgb_encoder.normalize(rgb)
        rgb_grid = self.rgb_encoder.cnn[:-1](rgb)
        rgb_grid = self.rgb_peg(rgb_grid)
        rgb_grid = F.adaptive_avg_pool2d(rgb_grid, (1, 1))
        return self.rgb_encoder.fc(rgb_grid)

    def _encode_depth_with_peg(self, depth_batch: Tensor) -> Tensor:
        depth_grid = self.depth_encoder.visual_encoder(depth_batch)
        depth_grid = self.depth_peg(depth_grid)
        return self.depth_encoder.fc(depth_grid)

    def forward(
        self,
        tokenized_instruction_batch: Tensor,
        depth_batch: Tensor,
        rgb_batch: Tensor,
        map_batch: Tensor,
        rnn_states_batch: Tensor,
        masks: Tensor,
    ):
        depth_batch = depth_batch.flip(-2)

        instruction_tokens = self.instruction_encoder(tokenized_instruction_batch).transpose(1, 2)
        instruction_tokens = self.instruction_proj(instruction_tokens)
        instruction_padding_mask = tokenized_instruction_batch[:, :instruction_tokens.size(1)].eq(0)
        instruction_valid_mask = ~instruction_padding_mask

        map_token = self.map_proj(self._encode_map_with_peg(map_batch)).unsqueeze(1)
        rgb_token = self.rgb_proj(self._encode_rgb_with_peg(rgb_batch)).unsqueeze(1)
        depth_token = self.depth_proj(self._encode_depth_with_peg(depth_batch)).unsqueeze(1)
        visual_tokens = torch.cat((map_token, rgb_token, depth_token), dim=1)
        visual_tokens = visual_tokens + self.visual_pos + self.modality_embed.weight.unsqueeze(0)

        memory = self.visual_norm(self.visual_encoder(visual_tokens))
        decoded_instruction = self.instruction_decoder(
            tgt=instruction_tokens,
            memory=memory,
            tgt_key_padding_mask=instruction_padding_mask,
        )
        summary = self.summary_norm(self._masked_mean(decoded_instruction, instruction_valid_mask))

        pred_normalized_goal_xys = self.goal_prediction_head(summary)
        pred_progress = self.progress_prediction_head(summary)
        return pred_normalized_goal_xys, pred_progress, rnn_states_batch

    def get_initial_recurrent_hidden_states(self, batch_size: int, device: str):
        return torch.zeros(batch_size, self.num_recurrent_layers, 1, device=device)


class InstructionQueryDecoderWithUniPEMap(nn.Module):
    """Instruction-query decoder with unified coordinate positional encoding.

    This keeps the original Ins-Dec interface to the visual encoder: map, RGB,
    and depth are still three visual tokens. The difference is that each
    modality is first represented as patch tokens, each patch receives the same
    sinusoidal position encoding in the normalized map coordinate system, and
    only then is pooled into one modality token.
    """

    def __init__(
        self,
        map_size: int,
        hidden_size: int = 512,
        num_heads: int = 8,
        encoder_layers: int = 2,
        decoder_layers: int = 2,
        dropout: float = 0.1,
        num_position_frequencies: int = 16,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_position_frequencies = num_position_frequencies

        self.instruction_encoder = InstructionEncoder(
            final_state_only=False,
            bidirectional=True,
            vocab_size=30522,
        )
        self.map_patch_encoder = MapPatchEncoder(map_size)
        self.rgb_encoder = TorchVisionResNet50().eval()
        self.depth_encoder = ResnetDepthEncoder().eval()

        rgb_patch_channels = self.rgb_encoder.fc[1].in_features
        depth_patch_channels = self.depth_encoder.visual_encoder.output_shape[0]
        self.map_patch_proj = nn.Linear(self.map_patch_encoder.out_channels, hidden_size)
        self.rgb_patch_proj = nn.Linear(rgb_patch_channels, hidden_size)
        self.depth_patch_proj = nn.Linear(depth_patch_channels, hidden_size)
        self.instruction_proj = nn.Linear(self.instruction_encoder.output_size, hidden_size)

        self.modality_embed = nn.Embedding(3, hidden_size)
        self.uni_pe_proj = nn.Sequential(
            nn.Linear(num_position_frequencies * 4, hidden_size),
            nn.LayerNorm(hidden_size),
        )
        freqs = (2.0 ** torch.arange(num_position_frequencies, dtype=torch.float32)) * math.pi
        self.register_buffer("position_frequencies", freqs, persistent=False)
        self.register_buffer("map_patch_centers", self._build_grid_coords(self.map_patch_encoder.grid_size), persistent=False)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.visual_encoder = nn.TransformerEncoder(encoder_layer, num_layers=encoder_layers)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.instruction_decoder = nn.TransformerDecoder(decoder_layer, num_layers=decoder_layers)
        self.visual_norm = nn.LayerNorm(hidden_size)
        self.summary_norm = nn.LayerNorm(hidden_size)

        self.goal_prediction_head = nn.Sequential(
            nn.Linear(hidden_size, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, 2), nn.Sigmoid(),
        )
        self.progress_prediction_head = nn.Sequential(
            nn.Linear(hidden_size, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, 1), nn.Sigmoid(),
        )
        self.train()

    @property
    def num_recurrent_layers(self) -> int:
        return 1

    @staticmethod
    def _build_grid_coords(grid_size: int) -> Tensor:
        lin = torch.linspace(0.0, 1.0, grid_size)
        yy, xx = torch.meshgrid(lin, lin, indexing="ij")
        return torch.stack((xx, yy), dim=-1).reshape(-1, 2)

    def _masked_mean(self, tokens: Tensor, valid_mask: Tensor) -> Tensor:
        weights = valid_mask.to(tokens.dtype).unsqueeze(-1)
        return (tokens * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)

    def _position_encoding(self, coords: Tensor) -> Tensor:
        coords = coords.clamp(0.0, 1.0)
        angles = coords.unsqueeze(-1) * self.position_frequencies.view(1, 1, -1)
        x_angles, y_angles = angles.unbind(dim=2)
        pe = torch.cat((
            x_angles.sin(),
            x_angles.cos(),
            y_angles.sin(),
            y_angles.cos(),
        ), dim=-1)
        return self.uni_pe_proj(pe)

    def _view_bbox_coords(self, map_batch: Tensor, grid_size: int) -> Tensor:
        current_view = map_batch[:, 0].float() > 0
        batch_size, height, width = current_view.shape
        y_centers = torch.linspace(0.5 / height, 1.0 - 0.5 / height, height, device=map_batch.device)
        x_centers = torch.linspace(0.5 / width, 1.0 - 0.5 / width, width, device=map_batch.device)

        col_has_view = current_view.any(dim=1)
        row_has_view = current_view.any(dim=2)
        valid = current_view.flatten(1).any(dim=1)
        x_min = torch.where(col_has_view, x_centers.unsqueeze(0), torch.ones((), device=map_batch.device)).min(dim=1).values
        x_max = torch.where(col_has_view, x_centers.unsqueeze(0), torch.zeros((), device=map_batch.device)).max(dim=1).values
        y_min = torch.where(row_has_view, y_centers.unsqueeze(0), torch.ones((), device=map_batch.device)).min(dim=1).values
        y_max = torch.where(row_has_view, y_centers.unsqueeze(0), torch.zeros((), device=map_batch.device)).max(dim=1).values

        x_min = torch.where(valid, x_min, torch.zeros_like(x_min))
        x_max = torch.where(valid, x_max, torch.ones_like(x_max))
        y_min = torch.where(valid, y_min, torch.zeros_like(y_min))
        y_max = torch.where(valid, y_max, torch.ones_like(y_max))

        patch_grid = self._build_grid_coords(grid_size).to(map_batch.device)
        x = x_min.unsqueeze(1) + patch_grid[:, 0].unsqueeze(0) * (x_max - x_min).clamp_min(1e-3).unsqueeze(1)
        y = y_min.unsqueeze(1) + patch_grid[:, 1].unsqueeze(0) * (y_max - y_min).clamp_min(1e-3).unsqueeze(1)
        return torch.stack((x, y), dim=-1)

    def _encode_rgb_patches(self, rgb_batch: Tensor) -> Tensor:
        rgb = rgb_batch.contiguous() / 255.0
        rgb = self.rgb_encoder.normalize(rgb)
        feature_grid = self.rgb_encoder.cnn[:-1](rgb)
        return feature_grid.flatten(2).transpose(1, 2)

    def _encode_depth_patches(self, depth_batch: Tensor) -> Tensor:
        feature_grid = self.depth_encoder.visual_encoder(depth_batch)
        return feature_grid.flatten(2).transpose(1, 2)

    def forward(
        self,
        tokenized_instruction_batch: Tensor,
        depth_batch: Tensor,
        rgb_batch: Tensor,
        map_batch: Tensor,
        rnn_states_batch: Tensor,
        masks: Tensor,
    ):
        depth_batch = depth_batch.flip(-2)

        instruction_tokens = self.instruction_encoder(tokenized_instruction_batch).transpose(1, 2)
        instruction_tokens = self.instruction_proj(instruction_tokens)
        instruction_padding_mask = tokenized_instruction_batch[:, :instruction_tokens.size(1)].eq(0)
        instruction_valid_mask = ~instruction_padding_mask

        batch_size = map_batch.size(0)
        map_tokens = self.map_patch_proj(self.map_patch_encoder(map_batch))
        map_coords = self.map_patch_centers.to(map_batch.device).unsqueeze(0).expand(batch_size, -1, -1)
        map_tokens = map_tokens + self._position_encoding(map_coords) + self.modality_embed.weight[0].view(1, 1, -1)

        rgb_tokens = self.rgb_patch_proj(self._encode_rgb_patches(rgb_batch))
        rgb_grid_size = int(math.sqrt(rgb_tokens.size(1)))
        rgb_coords = self._view_bbox_coords(map_batch, rgb_grid_size)
        rgb_tokens = rgb_tokens + self._position_encoding(rgb_coords) + self.modality_embed.weight[1].view(1, 1, -1)

        depth_tokens = self.depth_patch_proj(self._encode_depth_patches(depth_batch))
        depth_grid_size = int(math.sqrt(depth_tokens.size(1)))
        depth_coords = self._view_bbox_coords(map_batch, depth_grid_size)
        depth_tokens = depth_tokens + self._position_encoding(depth_coords) + self.modality_embed.weight[2].view(1, 1, -1)

        map_token = map_tokens.mean(dim=1, keepdim=True)
        rgb_token = rgb_tokens.mean(dim=1, keepdim=True)
        depth_token = depth_tokens.mean(dim=1, keepdim=True)
        visual_tokens = torch.cat((map_token, rgb_token, depth_token), dim=1)
        memory = self.visual_norm(self.visual_encoder(visual_tokens))
        decoded_instruction = self.instruction_decoder(
            tgt=instruction_tokens,
            memory=memory,
            tgt_key_padding_mask=instruction_padding_mask,
        )
        summary = self.summary_norm(self._masked_mean(decoded_instruction, instruction_valid_mask))

        pred_normalized_goal_xys = self.goal_prediction_head(summary)
        pred_progress = self.progress_prediction_head(summary)
        return pred_normalized_goal_xys, pred_progress, rnn_states_batch

    def get_initial_recurrent_hidden_states(self, batch_size: int, device: str):
        return torch.zeros(batch_size, self.num_recurrent_layers, 1, device=device)


class DilutedMasaRgbMapAttention(nn.Module):
    """RGB-to-map cross attention with softened spatial weights.

    Spatial weighting follows A' = A * (1 + alpha * (D - 1)), but D is first
    diluted toward its row mean: D <- (D + mean(D)) / 2. This keeps the spatial
    prior from overwhelming learned attention early in training.
    """

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
        self.distance_scale = nn.Parameter(torch.tensor(0.5))

    def forward(self, rgb_token: Tensor, map_tokens: Tensor, distances: Tensor) -> Tensor:
        batch_size, _, _ = rgb_token.shape
        num_map_tokens = map_tokens.size(1)
        q = self.q_proj(rgb_token).view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(map_tokens).view(batch_size, num_map_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(map_tokens).view(batch_size, num_map_tokens, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attention = scores.softmax(dim=-1)

        spatial_weight = torch.exp(-distances / F.softplus(self.distance_scale).clamp_min(1e-4))
        spatial_weight = 0.5 * (spatial_weight + spatial_weight.mean(dim=-1, keepdim=True))
        alpha = torch.sigmoid(self.alpha_logit)
        attention = attention * (1.0 + alpha * (spatial_weight.unsqueeze(1) - 1.0))
        attention = attention.clamp_min(1e-6)
        attention = attention / attention.sum(dim=-1, keepdim=True)
        attention = self.dropout(attention)

        attended = torch.matmul(attention, v).transpose(1, 2).reshape(batch_size, 1, self.hidden_size)
        return self.out_proj(attended)


class DilutedMasaRgbMapBlock(nn.Module):

    def __init__(self, hidden_size: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.rgb_norm = nn.LayerNorm(hidden_size)
        self.map_norm = nn.LayerNorm(hidden_size)
        self.attn = DilutedMasaRgbMapAttention(hidden_size, num_heads, dropout)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, hidden_size),
            nn.Dropout(dropout),
        )

    def forward(self, rgb_token: Tensor, map_tokens: Tensor, distances: Tensor) -> Tensor:
        rgb_token = rgb_token + self.attn(self.rgb_norm(rgb_token), self.map_norm(map_tokens), distances)
        rgb_token = rgb_token + self.ffn(self.norm2(rgb_token))
        return rgb_token


class UnifiedSpatialConstraintAttention(nn.Module):
    """RGB-map cross attention constrained in a shared global coordinate frame."""

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

    def forward(self, rgb_tokens: Tensor, map_tokens: Tensor, distances: Tensor) -> Tensor:
        batch_size, num_rgb_tokens, _ = rgb_tokens.shape
        num_map_tokens = map_tokens.size(1)

        q = self.q_proj(rgb_tokens).view(batch_size, num_rgb_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(map_tokens).view(batch_size, num_map_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(map_tokens).view(batch_size, num_map_tokens, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attention = scores.softmax(dim=-1)

        spatial_weight = torch.exp(-distances / F.softplus(self.distance_scale).clamp_min(1e-4))
        spatial_weight = 0.5 * (spatial_weight + spatial_weight.mean(dim=-1, keepdim=True))
        alpha = torch.sigmoid(self.alpha_logit)
        attention = attention * (1.0 + alpha * (spatial_weight.unsqueeze(1) - 1.0))
        attention = attention.clamp_min(1e-6)
        attention = attention / attention.sum(dim=-1, keepdim=True)
        attention = self.dropout(attention)

        attended = torch.matmul(attention, v).transpose(1, 2).reshape(batch_size, num_rgb_tokens, self.hidden_size)
        return self.out_proj(attended)


class UnifiedSpatialConstraintBlock(nn.Module):

    def __init__(self, hidden_size: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.rgb_norm = nn.LayerNorm(hidden_size)
        self.map_norm = nn.LayerNorm(hidden_size)
        self.attn = UnifiedSpatialConstraintAttention(hidden_size, num_heads, dropout)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, hidden_size),
            nn.Dropout(dropout),
        )

    def forward(self, rgb_tokens: Tensor, map_tokens: Tensor, distances: Tensor) -> Tensor:
        rgb_tokens = rgb_tokens + self.attn(self.rgb_norm(rgb_tokens), self.map_norm(map_tokens), distances)
        rgb_tokens = rgb_tokens + self.ffn(self.norm2(rgb_tokens))
        return rgb_tokens


class InstructionKVTokenCrossAttention(nn.Module):
    """Cross-attention that uses visual tokens as queries and instruction tokens as K/V."""

    def __init__(self, hidden_size: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.query_norm = nn.LayerNorm(hidden_size)
        self.instruction_norm = nn.LayerNorm(hidden_size)
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, hidden_size),
            nn.Dropout(dropout),
        )

    def forward(self, query_tokens: Tensor, instruction_tokens: Tensor, instruction_padding_mask: Tensor) -> Tensor:
        attended, _ = self.attn(
            query=self.query_norm(query_tokens),
            key=self.instruction_norm(instruction_tokens),
            value=self.instruction_norm(instruction_tokens),
            key_padding_mask=instruction_padding_mask,
            need_weights=False,
        )
        query_tokens = query_tokens + attended
        query_tokens = query_tokens + self.ffn(self.norm2(query_tokens))
        return query_tokens


class InstructionQueryDecoderWithUSCMap(nn.Module):
    """Ins-Dec with unified spatial constraints between RGB patches and map patches.

    The main Ins-Dec architecture is unchanged: map, RGB, and depth are pooled
    into three visual memory tokens, encoded, then queried by instruction
    embeddings. USC only refines the RGB token through cross-attention from RGB
    feature-grid tokens to map patch tokens in a shared normalized map frame.
    """

    def __init__(
        self,
        map_size: int,
        hidden_size: int = 512,
        num_heads: int = 8,
        usc_layers: int = 1,
        encoder_layers: int = 2,
        decoder_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_size = hidden_size

        self.instruction_encoder = InstructionEncoder(
            final_state_only=False,
            bidirectional=True,
            vocab_size=30522,
        )
        self.map_encoder = MapEncoder(map_size)
        self.map_patch_encoder = MapPatchEncoder(map_size)
        self.rgb_encoder = TorchVisionResNet50().eval()
        self.depth_encoder = ResnetDepthEncoder().eval()

        rgb_patch_channels = self.rgb_encoder.fc[1].in_features
        self.map_proj = nn.Linear(self.map_encoder.out_features, hidden_size)
        self.map_patch_proj = nn.Linear(self.map_patch_encoder.out_channels, hidden_size)
        self.rgb_proj = nn.Linear(self.rgb_encoder.out_features, hidden_size)
        self.rgb_patch_proj = nn.Linear(rgb_patch_channels, hidden_size)
        self.depth_proj = nn.Linear(self.depth_encoder.out_features, hidden_size)
        self.instruction_proj = nn.Linear(self.instruction_encoder.output_size, hidden_size)

        self.visual_pos = nn.Parameter(torch.zeros(1, 3, hidden_size))
        self.modality_embed = nn.Embedding(3, hidden_size)
        self.map_patch_norm = nn.LayerNorm(hidden_size)
        self.rgb_patch_norm = nn.LayerNorm(hidden_size)
        self.usc_blocks = nn.ModuleList([
            UnifiedSpatialConstraintBlock(hidden_size, num_heads, dropout)
            for _ in range(usc_layers)
        ])
        self.usc_pool = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
        )
        self.usc_residual_gate = nn.Parameter(torch.tensor(-2.0))
        self.register_buffer("map_patch_centers", self._build_grid_coords(self.map_patch_encoder.grid_size), persistent=False)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.visual_encoder = nn.TransformerEncoder(encoder_layer, num_layers=encoder_layers)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.instruction_decoder = nn.TransformerDecoder(decoder_layer, num_layers=decoder_layers)
        self.visual_norm = nn.LayerNorm(hidden_size)
        self.summary_norm = nn.LayerNorm(hidden_size)

        self.goal_prediction_head = nn.Sequential(
            nn.Linear(hidden_size, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, 2), nn.Sigmoid(),
        )
        self.progress_prediction_head = nn.Sequential(
            nn.Linear(hidden_size, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, 1), nn.Sigmoid(),
        )

        nn.init.trunc_normal_(self.visual_pos, std=0.02)
        self.train()

    @property
    def num_recurrent_layers(self) -> int:
        return 1

    @staticmethod
    def _build_grid_coords(grid_size: int) -> Tensor:
        lin = torch.linspace(0.0, 1.0, grid_size)
        yy, xx = torch.meshgrid(lin, lin, indexing="ij")
        return torch.stack((xx, yy), dim=-1).reshape(-1, 2)

    def _masked_mean(self, tokens: Tensor, valid_mask: Tensor) -> Tensor:
        weights = valid_mask.to(tokens.dtype).unsqueeze(-1)
        return (tokens * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)

    def _view_bbox_coords(self, map_batch: Tensor, grid_size: int) -> Tensor:
        current_view = map_batch[:, 0].float() > 0
        batch_size, height, width = current_view.shape
        y_centers = torch.linspace(0.5 / height, 1.0 - 0.5 / height, height, device=map_batch.device)
        x_centers = torch.linspace(0.5 / width, 1.0 - 0.5 / width, width, device=map_batch.device)

        col_has_view = current_view.any(dim=1)
        row_has_view = current_view.any(dim=2)
        valid = current_view.flatten(1).any(dim=1)
        x_min = torch.where(col_has_view, x_centers.unsqueeze(0), torch.ones((), device=map_batch.device)).min(dim=1).values
        x_max = torch.where(col_has_view, x_centers.unsqueeze(0), torch.zeros((), device=map_batch.device)).max(dim=1).values
        y_min = torch.where(row_has_view, y_centers.unsqueeze(0), torch.ones((), device=map_batch.device)).min(dim=1).values
        y_max = torch.where(row_has_view, y_centers.unsqueeze(0), torch.zeros((), device=map_batch.device)).max(dim=1).values

        x_min = torch.where(valid, x_min, torch.zeros_like(x_min))
        x_max = torch.where(valid, x_max, torch.ones_like(x_max))
        y_min = torch.where(valid, y_min, torch.zeros_like(y_min))
        y_max = torch.where(valid, y_max, torch.ones_like(y_max))

        patch_grid = self._build_grid_coords(grid_size).to(map_batch.device)
        x = x_min.unsqueeze(1) + patch_grid[:, 0].unsqueeze(0) * (x_max - x_min).clamp_min(1e-3).unsqueeze(1)
        y = y_min.unsqueeze(1) + patch_grid[:, 1].unsqueeze(0) * (y_max - y_min).clamp_min(1e-3).unsqueeze(1)
        return torch.stack((x, y), dim=-1)

    def _encode_rgb_grid(self, rgb_batch: Tensor) -> Tensor:
        rgb = rgb_batch.contiguous() / 255.0
        rgb = self.rgb_encoder.normalize(rgb)
        return self.rgb_encoder.cnn[:-1](rgb)

    def forward(
        self,
        tokenized_instruction_batch: Tensor,
        depth_batch: Tensor,
        rgb_batch: Tensor,
        map_batch: Tensor,
        rnn_states_batch: Tensor,
        masks: Tensor,
    ):
        depth_batch = depth_batch.flip(-2)

        instruction_tokens = self.instruction_encoder(tokenized_instruction_batch).transpose(1, 2)
        instruction_tokens = self.instruction_proj(instruction_tokens)
        instruction_padding_mask = tokenized_instruction_batch[:, :instruction_tokens.size(1)].eq(0)
        instruction_valid_mask = ~instruction_padding_mask

        batch_size = map_batch.size(0)
        map_token = self.map_proj(self.map_encoder(map_batch)).unsqueeze(1)
        map_patch_tokens = self.map_patch_proj(self.map_patch_encoder(map_batch))
        map_patch_tokens = map_patch_tokens + self.modality_embed.weight[0].view(1, 1, -1)
        map_patch_tokens = self.map_patch_norm(map_patch_tokens)
        map_coords = self.map_patch_centers.to(map_batch.device).unsqueeze(0).expand(batch_size, -1, -1)

        rgb_grid = self._encode_rgb_grid(rgb_batch)
        rgb_token = self.rgb_proj(self.rgb_encoder.fc(F.adaptive_avg_pool2d(rgb_grid, (1, 1)))).unsqueeze(1)
        rgb_patch_tokens = self.rgb_patch_proj(rgb_grid.flatten(2).transpose(1, 2))
        rgb_patch_tokens = rgb_patch_tokens + self.modality_embed.weight[1].view(1, 1, -1)
        rgb_patch_tokens = self.rgb_patch_norm(rgb_patch_tokens)
        rgb_grid_size = int(math.sqrt(rgb_patch_tokens.size(1)))
        rgb_coords = self._view_bbox_coords(map_batch, rgb_grid_size)
        rgb_map_distances = torch.cdist(rgb_coords, map_coords, p=2)

        for block in self.usc_blocks:
            rgb_patch_tokens = block(rgb_patch_tokens, map_patch_tokens, rgb_map_distances)
        usc_rgb_token = self.usc_pool(rgb_patch_tokens.mean(dim=1, keepdim=True))
        rgb_token = rgb_token + torch.sigmoid(self.usc_residual_gate) * (usc_rgb_token - rgb_token)

        depth_token = self.depth_proj(self.depth_encoder(depth_batch)).unsqueeze(1)
        
        visual_tokens = torch.cat((map_token, rgb_token, depth_token), dim=1)
        visual_tokens = visual_tokens + self.visual_pos + self.modality_embed.weight.unsqueeze(0)

        memory = self.visual_norm(self.visual_encoder(visual_tokens))
        decoded_instruction = self.instruction_decoder(
            tgt=instruction_tokens,
            memory=memory,
            tgt_key_padding_mask=instruction_padding_mask,
        )
        summary = self.summary_norm(self._masked_mean(decoded_instruction, instruction_valid_mask))

        pred_normalized_goal_xys = self.goal_prediction_head(summary)
        pred_progress = self.progress_prediction_head(summary)
        return pred_normalized_goal_xys, pred_progress, rnn_states_batch

    def get_initial_recurrent_hidden_states(self, batch_size: int, device: str):
        return torch.zeros(batch_size, self.num_recurrent_layers, 1, device=device)


class InstructionKVUSCMap(InstructionQueryDecoderWithUSCMap):
    """Instruction-KV cross attention before USC RGB-map fusion.

    Map and RGB patch tokens first query the instruction sequence in two
    independent cross-attention blocks. The instruction-enhanced map/RGB tokens
    are then fused by USC, and their pooled summaries are passed to the original
    Ins-Dec visual encoder, instruction decoder, and prediction heads.
    """

    def __init__(
        self,
        map_size: int,
        hidden_size: int = 512,
        num_heads: int = 8,
        usc_layers: int = 1,
        encoder_layers: int = 2,
        decoder_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__(map_size, hidden_size, num_heads, usc_layers, encoder_layers, decoder_layers, dropout)
        self.map_instruction_cross_attn = InstructionKVTokenCrossAttention(hidden_size, num_heads, dropout)
        self.rgb_instruction_cross_attn = InstructionKVTokenCrossAttention(hidden_size, num_heads, dropout)
        self.map_pool = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
        )
        self.rgb_pool = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
        )

    def forward(
        self,
        tokenized_instruction_batch: Tensor,
        depth_batch: Tensor,
        rgb_batch: Tensor,
        map_batch: Tensor,
        rnn_states_batch: Tensor,
        masks: Tensor,
    ):
        depth_batch = depth_batch.flip(-2)

        instruction_tokens = self.instruction_encoder(tokenized_instruction_batch).transpose(1, 2)
        instruction_tokens = self.instruction_proj(instruction_tokens)
        instruction_padding_mask = tokenized_instruction_batch[:, :instruction_tokens.size(1)].eq(0)
        instruction_valid_mask = ~instruction_padding_mask

        batch_size = map_batch.size(0)
        map_patch_tokens = self.map_patch_proj(self.map_patch_encoder(map_batch))
        map_patch_tokens = map_patch_tokens + self.modality_embed.weight[0].view(1, 1, -1)
        map_patch_tokens = self.map_patch_norm(map_patch_tokens)
        map_patch_tokens = self.map_instruction_cross_attn(
            map_patch_tokens,
            instruction_tokens,
            instruction_padding_mask,
        )
        map_coords = self.map_patch_centers.to(map_batch.device).unsqueeze(0).expand(batch_size, -1, -1)

        rgb_grid = self._encode_rgb_grid(rgb_batch)
        rgb_patch_tokens = self.rgb_patch_proj(rgb_grid.flatten(2).transpose(1, 2))
        rgb_patch_tokens = rgb_patch_tokens + self.modality_embed.weight[1].view(1, 1, -1)
        rgb_patch_tokens = self.rgb_patch_norm(rgb_patch_tokens)
        rgb_patch_tokens = self.rgb_instruction_cross_attn(
            rgb_patch_tokens,
            instruction_tokens,
            instruction_padding_mask,
        )
        rgb_grid_size = int(math.sqrt(rgb_patch_tokens.size(1)))
        rgb_coords = self._view_bbox_coords(map_batch, rgb_grid_size)
        rgb_map_distances = torch.cdist(rgb_coords, map_coords, p=2)

        for block in self.usc_blocks:
            rgb_patch_tokens = block(rgb_patch_tokens, map_patch_tokens, rgb_map_distances)

        map_token = self.map_pool(map_patch_tokens.mean(dim=1, keepdim=True))
        rgb_token = self.rgb_pool(rgb_patch_tokens.mean(dim=1, keepdim=True))
        depth_token = self.depth_proj(self.depth_encoder(depth_batch)).unsqueeze(1)
        visual_tokens = torch.cat((map_token, rgb_token, depth_token), dim=1)
        visual_tokens = visual_tokens + self.visual_pos + self.modality_embed.weight.unsqueeze(0)

        memory = self.visual_norm(self.visual_encoder(visual_tokens))
        decoded_instruction = self.instruction_decoder(
            tgt=instruction_tokens,
            memory=memory,
            tgt_key_padding_mask=instruction_padding_mask,
        )
        summary = self.summary_norm(self._masked_mean(decoded_instruction, instruction_valid_mask))

        pred_normalized_goal_xys = self.goal_prediction_head(summary)
        pred_progress = self.progress_prediction_head(summary)
        return pred_normalized_goal_xys, pred_progress, rnn_states_batch


class InstructionQueryDecoderWithDirectionalUSCMap(InstructionQueryDecoderWithUSCMap):
    """USC with an RGB orientation embedding estimated from the map footprint.

    The model keeps the same USC cross-attention and Ins-Dec backbone. It only
    adds a direction-aware embedding to RGB patch tokens before USC attention.
    Direction is inferred from the current-view footprint in map channel 0, so
    the data interface remains identical to the original CityNav pipeline.
    """

    def __init__(
        self,
        map_size: int,
        hidden_size: int = 512,
        num_heads: int = 8,
        usc_layers: int = 1,
        encoder_layers: int = 2,
        decoder_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__(map_size, hidden_size, num_heads, usc_layers, encoder_layers, decoder_layers, dropout)
        self.rgb_direction_proj = nn.Sequential(
            nn.Linear(4, hidden_size),
            nn.LayerNorm(hidden_size),
        )

    def _estimate_view_direction(self, map_batch: Tensor) -> Tensor:
        current_view = map_batch[:, 0].float()
        explored_area = map_batch[:, 1].float()
        batch_size, height, width = current_view.shape
        y = torch.linspace(0.5 / height, 1.0 - 0.5 / height, height, device=map_batch.device)
        x = torch.linspace(0.5 / width, 1.0 - 0.5 / width, width, device=map_batch.device)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        coords = torch.stack((xx, yy), dim=-1).view(1, height * width, 2)

        view_weights = current_view.flatten(1)
        view_mass = view_weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
        view_center = (view_weights.unsqueeze(-1) * coords).sum(dim=1) / view_mass

        centered = coords - view_center.unsqueeze(1)
        cov = torch.matmul((centered * view_weights.unsqueeze(-1)).transpose(1, 2), centered) / view_mass.unsqueeze(-1)
        eigvals, eigvecs = torch.linalg.eigh(cov)
        direction = eigvecs[:, :, -1]

        explored_weights = explored_area.flatten(1)
        explored_mass = explored_weights.sum(dim=1, keepdim=True)
        explored_center = (explored_weights.unsqueeze(-1) * coords).sum(dim=1) / explored_mass.clamp_min(1e-6)
        sign_ref = view_center - explored_center
        sign = torch.sign((direction * sign_ref).sum(dim=-1, keepdim=True)).clamp(min=0.0) * 2.0 - 1.0
        direction = F.normalize(direction * sign, dim=-1, eps=1e-6)

        fallback = current_view.new_tensor([1.0, 0.0]).expand(batch_size, 2)
        valid = (view_weights.sum(dim=1, keepdim=True) > 0)
        return torch.where(valid, direction, fallback)

    def _rgb_direction_embedding(self, map_batch: Tensor, rgb_coords: Tensor) -> Tensor:
        direction = self._estimate_view_direction(map_batch)
        current_view = map_batch[:, 0].float()
        batch_size, height, width = current_view.shape
        y = torch.linspace(0.5 / height, 1.0 - 0.5 / height, height, device=map_batch.device)
        x = torch.linspace(0.5 / width, 1.0 - 0.5 / width, width, device=map_batch.device)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        coords = torch.stack((xx, yy), dim=-1).view(1, height * width, 2)
        weights = current_view.flatten(1)
        center = (weights.unsqueeze(-1) * coords).sum(dim=1) / weights.sum(dim=1, keepdim=True).clamp_min(1e-6)

        forward = direction.unsqueeze(1)
        right = torch.stack((direction[:, 1], -direction[:, 0]), dim=-1).unsqueeze(1)
        relative = rgb_coords - center.unsqueeze(1)
        local_x = (relative * right).sum(dim=-1, keepdim=True)
        local_y = (relative * forward).sum(dim=-1, keepdim=True)
        direction_features = torch.cat((
            direction.unsqueeze(1).expand(-1, rgb_coords.size(1), -1),
            local_x,
            local_y,
        ), dim=-1)
        return self.rgb_direction_proj(direction_features)

    def forward(
        self,
        tokenized_instruction_batch: Tensor,
        depth_batch: Tensor,
        rgb_batch: Tensor,
        map_batch: Tensor,
        rnn_states_batch: Tensor,
        masks: Tensor,
    ):
        depth_batch = depth_batch.flip(-2)

        instruction_tokens = self.instruction_encoder(tokenized_instruction_batch).transpose(1, 2)
        instruction_tokens = self.instruction_proj(instruction_tokens)
        instruction_padding_mask = tokenized_instruction_batch[:, :instruction_tokens.size(1)].eq(0)
        instruction_valid_mask = ~instruction_padding_mask

        batch_size = map_batch.size(0)
        map_token = self.map_proj(self.map_encoder(map_batch)).unsqueeze(1)
        map_patch_tokens = self.map_patch_proj(self.map_patch_encoder(map_batch))
        map_patch_tokens = map_patch_tokens + self.modality_embed.weight[0].view(1, 1, -1)
        map_patch_tokens = self.map_patch_norm(map_patch_tokens)
        map_coords = self.map_patch_centers.to(map_batch.device).unsqueeze(0).expand(batch_size, -1, -1)

        rgb_grid = self._encode_rgb_grid(rgb_batch)
        rgb_token = self.rgb_proj(self.rgb_encoder.fc(F.adaptive_avg_pool2d(rgb_grid, (1, 1)))).unsqueeze(1)
        rgb_patch_tokens = self.rgb_patch_proj(rgb_grid.flatten(2).transpose(1, 2))
        rgb_grid_size = int(math.sqrt(rgb_patch_tokens.size(1)))
        rgb_coords = self._view_bbox_coords(map_batch, rgb_grid_size)
        rgb_patch_tokens = rgb_patch_tokens + self.modality_embed.weight[1].view(1, 1, -1)
        rgb_patch_tokens = rgb_patch_tokens + self._rgb_direction_embedding(map_batch, rgb_coords)
        rgb_patch_tokens = self.rgb_patch_norm(rgb_patch_tokens)
        rgb_map_distances = torch.cdist(rgb_coords, map_coords, p=2)

        for block in self.usc_blocks:
            rgb_patch_tokens = block(rgb_patch_tokens, map_patch_tokens, rgb_map_distances)
        usc_rgb_token = self.usc_pool(rgb_patch_tokens.mean(dim=1, keepdim=True))
        rgb_token = rgb_token + torch.sigmoid(self.usc_residual_gate) * (usc_rgb_token - rgb_token)

        depth_token = self.depth_proj(self.depth_encoder(depth_batch)).unsqueeze(1)
        visual_tokens = torch.cat((map_token, rgb_token, depth_token), dim=1)
        visual_tokens = visual_tokens + self.visual_pos + self.modality_embed.weight.unsqueeze(0)

        memory = self.visual_norm(self.visual_encoder(visual_tokens))
        decoded_instruction = self.instruction_decoder(
            tgt=instruction_tokens,
            memory=memory,
            tgt_key_padding_mask=instruction_padding_mask,
        )
        summary = self.summary_norm(self._masked_mean(decoded_instruction, instruction_valid_mask))

        pred_normalized_goal_xys = self.goal_prediction_head(summary)
        pred_progress = self.progress_prediction_head(summary)
        return pred_normalized_goal_xys, pred_progress, rnn_states_batch


class InstructionQueryDecoderWithDilutedMasaMap(nn.Module):
    """Instruction-query decoder whose map memory is refined by diluted MaSA."""

    def __init__(
        self,
        map_size: int,
        hidden_size: int = 512,
        num_heads: int = 8,
        masa_layers: int = 2,
        encoder_layers: int = 2,
        decoder_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_size = hidden_size

        self.instruction_encoder = InstructionEncoder(
            final_state_only=False,
            bidirectional=True,
            vocab_size=30522,
        )
        self.map_encoder = MapEncoder(map_size)
        self.map_patch_encoder = MapPatchEncoder(map_size)
        self.rgb_encoder = TorchVisionResNet50().eval()
        self.depth_encoder = ResnetDepthEncoder().eval()

        self.map_proj = nn.Linear(self.map_encoder.out_features, hidden_size)
        self.map_patch_proj = nn.Linear(self.map_patch_encoder.out_channels, hidden_size)
        self.rgb_proj = nn.Linear(self.rgb_encoder.out_features, hidden_size)
        self.depth_proj = nn.Linear(self.depth_encoder.out_features, hidden_size)
        self.instruction_proj = nn.Linear(self.instruction_encoder.output_size, hidden_size)

        num_map_patch_tokens = self.map_patch_encoder.grid_size**2
        self.map_patch_pos = nn.Parameter(torch.zeros(1, num_map_patch_tokens, hidden_size))
        self.visual_pos = nn.Parameter(torch.zeros(1, 3, hidden_size))
        self.modality_embed = nn.Embedding(3, hidden_size)
        coords = self._build_grid_coords(self.map_patch_encoder.grid_size)
        self.register_buffer("map_patch_centers", coords, persistent=False)

        self.rgb_map_masa_blocks = nn.ModuleList([
            DilutedMasaRgbMapBlock(hidden_size, num_heads, dropout)
            for _ in range(masa_layers)
        ])
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.visual_encoder = nn.TransformerEncoder(encoder_layer, num_layers=encoder_layers)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.instruction_decoder = nn.TransformerDecoder(decoder_layer, num_layers=decoder_layers)
        self.map_norm = nn.LayerNorm(hidden_size)
        self.visual_norm = nn.LayerNorm(hidden_size)
        self.summary_norm = nn.LayerNorm(hidden_size)

        self.goal_prediction_head = nn.Sequential(
            nn.Linear(hidden_size, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, 2), nn.Sigmoid(),
        )
        self.progress_prediction_head = nn.Sequential(
            nn.Linear(hidden_size, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, 1), nn.Sigmoid(),
        )

        nn.init.trunc_normal_(self.map_patch_pos, std=0.02)
        nn.init.trunc_normal_(self.visual_pos, std=0.02)
        self.train()

    @property
    def num_recurrent_layers(self) -> int:
        return 1

    @staticmethod
    def _build_grid_coords(grid_size: int) -> Tensor:
        lin = torch.linspace(0.0, 1.0, grid_size)
        yy, xx = torch.meshgrid(lin, lin, indexing="ij")
        return torch.stack((xx, yy), dim=-1).reshape(-1, 2)

    def _masked_mean(self, tokens: Tensor, valid_mask: Tensor) -> Tensor:
        weights = valid_mask.to(tokens.dtype).unsqueeze(-1)
        return (tokens * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)

    def _estimate_rgb_coords(self, map_batch: Tensor) -> Tensor:
        """Estimate current RGB/global position from tracking channel center.

        Channel 0 is the current view area. Coordinates are normalized to the
        map frame, with patch centers using the same [0, 1] x [0, 1] system.
        """
        current_view = map_batch[:, 0].float()
        batch_size, height, width = current_view.shape
        y_coords = torch.linspace(0.5 / height, 1.0 - 0.5 / height, height, device=map_batch.device)
        x_coords = torch.linspace(0.5 / width, 1.0 - 0.5 / width, width, device=map_batch.device)
        mass = current_view.flatten(1).sum(dim=1, keepdim=True)
        default = current_view.new_tensor([0.5, 0.5]).expand(batch_size, 2)
        x = (current_view.sum(dim=1) * x_coords.unsqueeze(0)).sum(dim=1, keepdim=True)
        y = (current_view.sum(dim=2) * y_coords.unsqueeze(0)).sum(dim=1, keepdim=True)
        coords = torch.cat((x, y), dim=1) / mass.clamp_min(1e-6)
        return torch.where(mass > 0, coords, default)

    def forward(
        self,
        tokenized_instruction_batch: Tensor,
        depth_batch: Tensor,
        rgb_batch: Tensor,
        map_batch: Tensor,
        rnn_states_batch: Tensor,
        masks: Tensor,
    ):
        depth_batch = depth_batch.flip(-2)

        instruction_tokens = self.instruction_encoder(tokenized_instruction_batch).transpose(1, 2)
        instruction_tokens = self.instruction_proj(instruction_tokens)
        instruction_padding_mask = tokenized_instruction_batch[:, :instruction_tokens.size(1)].eq(0)
        instruction_valid_mask = ~instruction_padding_mask

        map_token = self.map_proj(self.map_encoder(map_batch)).unsqueeze(1)
        map_patch_tokens = self.map_patch_proj(self.map_patch_encoder(map_batch))
        map_patch_tokens = map_patch_tokens + self.map_patch_pos + self.modality_embed.weight[0].view(1, 1, -1)
        map_patch_tokens = self.map_norm(map_patch_tokens)

        rgb_token = self.rgb_proj(self.rgb_encoder(rgb_batch)).unsqueeze(1)
        rgb_coords = self._estimate_rgb_coords(map_batch)
        rgb_map_distances = torch.cdist(rgb_coords.unsqueeze(1), self.map_patch_centers.unsqueeze(0), p=2)
        for block in self.rgb_map_masa_blocks:
            rgb_token = block(rgb_token, map_patch_tokens, rgb_map_distances)

        depth_token = self.depth_proj(self.depth_encoder(depth_batch)).unsqueeze(1)
        visual_tokens = torch.cat((map_token, rgb_token, depth_token), dim=1)
        visual_tokens = visual_tokens + self.visual_pos + self.modality_embed.weight.unsqueeze(0)
        memory = self.visual_norm(self.visual_encoder(visual_tokens))
        decoded_instruction = self.instruction_decoder(
            tgt=instruction_tokens,
            memory=memory,
            tgt_key_padding_mask=instruction_padding_mask,
        )
        summary = self.summary_norm(self._masked_mean(decoded_instruction, instruction_valid_mask))

        pred_normalized_goal_xys = self.goal_prediction_head(summary)
        pred_progress = self.progress_prediction_head(summary)
        return pred_normalized_goal_xys, pred_progress, rnn_states_batch

    def get_initial_recurrent_hidden_states(self, batch_size: int, device: str):
        return torch.zeros(batch_size, self.num_recurrent_layers, 1, device=device)


class InstructionQueryDecoderWithResidualMasaMap(InstructionQueryDecoderWithDilutedMasaMap):
    """Diluted MaSA with a learnable residual gate on the RGB token.

    This keeps the instruction decoder and all non-MaSA attention identical to
    `InstructionQueryDecoderWithMap`. MaSA proposes an RGB-map refined token,
    but the model initially receives only a small residual update:

        rgb = rgb_clean + sigmoid(gate) * (rgb_masa - rgb_clean)

    The gate is scalar and initialized small so goal localization can fall back
    to the original RGB embedding while training decides whether MaSA is useful.
    """

    def __init__(
        self,
        map_size: int,
        hidden_size: int = 512,
        num_heads: int = 8,
        masa_layers: int = 2,
        encoder_layers: int = 2,
        decoder_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__(map_size, hidden_size, num_heads, masa_layers, encoder_layers, decoder_layers, dropout)
        self.masa_residual_gate = nn.Parameter(torch.tensor(-3.0))

    def _masa_rgb_token(self, rgb_token: Tensor, map_batch: Tensor, map_patch_tokens: Tensor) -> Tensor:
        rgb_coords = self._estimate_rgb_coords(map_batch)
        rgb_map_distances = torch.cdist(rgb_coords.unsqueeze(1), self.map_patch_centers.unsqueeze(0), p=2)
        masa_rgb_token = rgb_token
        for block in self.rgb_map_masa_blocks:
            masa_rgb_token = block(masa_rgb_token, map_patch_tokens, rgb_map_distances)
        return masa_rgb_token

    def _decode_summary(
        self,
        instruction_tokens: Tensor,
        instruction_padding_mask: Tensor,
        instruction_valid_mask: Tensor,
        visual_tokens: Tensor,
    ) -> Tensor:
        visual_tokens = visual_tokens + self.visual_pos + self.modality_embed.weight.unsqueeze(0)
        memory = self.visual_norm(self.visual_encoder(visual_tokens))
        decoded_instruction = self.instruction_decoder(
            tgt=instruction_tokens,
            memory=memory,
            tgt_key_padding_mask=instruction_padding_mask,
        )
        return self.summary_norm(self._masked_mean(decoded_instruction, instruction_valid_mask))

    def forward(
        self,
        tokenized_instruction_batch: Tensor,
        depth_batch: Tensor,
        rgb_batch: Tensor,
        map_batch: Tensor,
        rnn_states_batch: Tensor,
        masks: Tensor,
    ):
        depth_batch = depth_batch.flip(-2)

        instruction_tokens = self.instruction_encoder(tokenized_instruction_batch).transpose(1, 2)
        instruction_tokens = self.instruction_proj(instruction_tokens)
        instruction_padding_mask = tokenized_instruction_batch[:, :instruction_tokens.size(1)].eq(0)
        instruction_valid_mask = ~instruction_padding_mask

        map_token = self.map_proj(self.map_encoder(map_batch)).unsqueeze(1)
        map_patch_tokens = self.map_patch_proj(self.map_patch_encoder(map_batch))
        map_patch_tokens = map_patch_tokens + self.map_patch_pos + self.modality_embed.weight[0].view(1, 1, -1)
        map_patch_tokens = self.map_norm(map_patch_tokens)

        rgb_token = self.rgb_proj(self.rgb_encoder(rgb_batch)).unsqueeze(1)
        masa_rgb_token = self._masa_rgb_token(rgb_token, map_batch, map_patch_tokens)
        rgb_token = rgb_token + torch.sigmoid(self.masa_residual_gate) * (masa_rgb_token - rgb_token)

        depth_token = self.depth_proj(self.depth_encoder(depth_batch)).unsqueeze(1)
        visual_tokens = torch.cat((map_token, rgb_token, depth_token), dim=1)
        summary = self._decode_summary(
            instruction_tokens,
            instruction_padding_mask,
            instruction_valid_mask,
            visual_tokens,
        )

        pred_normalized_goal_xys = self.goal_prediction_head(summary)
        pred_progress = self.progress_prediction_head(summary)
        return pred_normalized_goal_xys, pred_progress, rnn_states_batch


class InstructionQueryDecoderWithProgressOnlyMasaMap(InstructionQueryDecoderWithResidualMasaMap):
    """Use MaSA only for progress prediction.

    The goal head receives the same three-token memory as the original
    instruction decoder. A separate MaSA-refined RGB token is decoded only for
    the progress head, preventing the spatial prior from perturbing goal
    localization while still testing its value for progress estimation.
    """

    def forward(
        self,
        tokenized_instruction_batch: Tensor,
        depth_batch: Tensor,
        rgb_batch: Tensor,
        map_batch: Tensor,
        rnn_states_batch: Tensor,
        masks: Tensor,
    ):
        depth_batch = depth_batch.flip(-2)

        instruction_tokens = self.instruction_encoder(tokenized_instruction_batch).transpose(1, 2)
        instruction_tokens = self.instruction_proj(instruction_tokens)
        instruction_padding_mask = tokenized_instruction_batch[:, :instruction_tokens.size(1)].eq(0)
        instruction_valid_mask = ~instruction_padding_mask

        map_token = self.map_proj(self.map_encoder(map_batch)).unsqueeze(1)
        map_patch_tokens = self.map_patch_proj(self.map_patch_encoder(map_batch))
        map_patch_tokens = map_patch_tokens + self.map_patch_pos + self.modality_embed.weight[0].view(1, 1, -1)
        map_patch_tokens = self.map_norm(map_patch_tokens)

        rgb_token = self.rgb_proj(self.rgb_encoder(rgb_batch)).unsqueeze(1)
        depth_token = self.depth_proj(self.depth_encoder(depth_batch)).unsqueeze(1)

        goal_visual_tokens = torch.cat((map_token, rgb_token, depth_token), dim=1)
        goal_summary = self._decode_summary(
            instruction_tokens,
            instruction_padding_mask,
            instruction_valid_mask,
            goal_visual_tokens,
        )

        masa_rgb_token = self._masa_rgb_token(rgb_token, map_batch, map_patch_tokens)
        progress_visual_tokens = torch.cat((map_token, masa_rgb_token, depth_token), dim=1)
        progress_summary = self._decode_summary(
            instruction_tokens,
            instruction_padding_mask,
            instruction_valid_mask,
            progress_visual_tokens,
        )

        pred_normalized_goal_xys = self.goal_prediction_head(goal_summary)
        pred_progress = self.progress_prediction_head(progress_summary)
        return pred_normalized_goal_xys, pred_progress, rnn_states_batch
