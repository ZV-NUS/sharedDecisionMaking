from __future__ import annotations

import math

import torch
from torch import nn


class HumanIntentTransformer(nn.Module):
    def __init__(
        self,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        history_len: int = 75,
        future_len: int = 125,
        num_decisions: int = 3,
    ) -> None:
        super().__init__()
        self.history_len = history_len
        self.future_len = future_len
        self.input_dim = 8 + 4 + 6 * 6 + 6

        self.input_proj = nn.Sequential(
            nn.Linear(self.input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.pos_encoding = SinusoidalPositionalEncoding(d_model, max_len=history_len)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pool = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
        )
        self.risk_interaction_dim = 4 * 3 + 1 + 6 + 6 + 6
        self.risk_interaction_proj = nn.Sequential(
            nn.Linear(self.risk_interaction_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.risk_interaction_gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.Sigmoid(),
        )
        self.event_context_norm = nn.LayerNorm(d_model)
        self.decision_head = nn.Linear(d_model, num_decisions)
        self.future_decision_head = nn.Linear(d_model, future_len * num_decisions)
        self.future_event_head = nn.Linear(d_model, num_decisions)
        self.future_event_time_head = nn.Linear(d_model, 1)
        self.future_event_time_bin_head = nn.Linear(d_model, future_len)
        self.future_event_time_by_class_head = nn.Linear(d_model, num_decisions)
        self.future_event_time_bin_by_class_head = nn.Linear(d_model, num_decisions * future_len)
        self.speed_head = nn.Linear(d_model, future_len)
        self.steer_head = nn.Linear(d_model, future_len)

    def forward(self, batch_inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        x = build_frame_features(batch_inputs)
        h = self.input_proj(x)
        h = self.pos_encoding(h)
        h = self.encoder(h)
        pooled = self.pool(h[:, -1])
        risk_interaction = self.risk_interaction_proj(build_risk_interaction_context(batch_inputs))
        event_gate = self.risk_interaction_gate(torch.cat([pooled, risk_interaction], dim=-1))
        event_context = self.event_context_norm(pooled + event_gate * risk_interaction)

        future_decision_logits = self.future_decision_head(pooled)
        future_decision_logits = future_decision_logits.view(
            pooled.shape[0],
            self.future_len,
            -1,
        )
        return {
            "decision_logits": self.decision_head(pooled),
            "future_decision_logits": future_decision_logits,
            "future_event_logits": self.future_event_head(event_context),
            "future_event_time": torch.sigmoid(self.future_event_time_head(event_context)).squeeze(-1),
            "future_event_time_logits": self.future_event_time_bin_head(event_context),
            "future_event_time_by_class": torch.sigmoid(self.future_event_time_by_class_head(event_context)),
            "future_event_time_bin_by_class": self.future_event_time_bin_by_class_head(event_context).view(
                pooled.shape[0],
                -1,
                self.future_len,
            ),
            "risk_interaction_gate": event_gate,
            "future_speed": batch_inputs["ego_history"][:, -1, 6:7] + self.speed_head(pooled),
            "future_steer": self.steer_head(pooled),
        }


def build_frame_features(batch_inputs: dict[str, torch.Tensor]) -> torch.Tensor:
    ego = batch_inputs["ego_history"]
    risk = batch_inputs["risk_history"]
    neighbors = batch_inputs["neighbor_history"].flatten(start_dim=2)
    mask = batch_inputs["neighbor_mask"]
    return torch.cat([ego, risk, neighbors, mask], dim=-1)


def build_risk_interaction_context(batch_inputs: dict[str, torch.Tensor]) -> torch.Tensor:
    risk = batch_inputs["risk_history"]
    risk_last = risk[:, -1]
    risk_mean = risk.mean(dim=1)
    risk_max = risk.amax(dim=1)

    neighbors = batch_inputs["neighbor_history"][:, -1]
    mask = batch_inputs["neighbor_mask"][:, -1]
    valid = mask.unsqueeze(-1).to(dtype=neighbors.dtype)
    valid_count = torch.clamp(valid.sum(dim=1), min=1.0)
    neighbor_mean = (neighbors * valid).sum(dim=1) / valid_count

    distances = torch.linalg.vector_norm(neighbors[..., :2], dim=-1)
    distances = distances.masked_fill(mask <= 0, 1.0e6)
    closest_idx = torch.argmin(distances, dim=1)
    closest = neighbors[torch.arange(neighbors.shape[0], device=neighbors.device), closest_idx]
    closest = closest * (mask.sum(dim=1, keepdim=True) > 0).to(dtype=neighbors.dtype)
    mask = mask.to(dtype=neighbors.dtype)
    valid_ratio = mask.mean(dim=1, keepdim=True)

    return torch.cat([risk_last, risk_mean, risk_max, valid_ratio, neighbor_mean, closest, mask], dim=-1)


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512) -> None:
        super().__init__()
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]
