from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn
import torch.nn.functional as F


@dataclass(frozen=True)
class TransformerSACConfig:
    obs_dim: int = 16
    future_len: int = 125
    d_model: int = 64
    nhead: int = 4
    num_layers: int = 2
    dim_feedforward: int = 128
    dropout: float = 0.1
    max_delta: float = 0.35
    actor_lr: float = 0.0003
    critic_lr: float = 0.0003
    alpha: float = 0.05
    log_std_min: float = -5.0
    log_std_max: float = 1.0


class TransformerSACAuthority(nn.Module):
    def __init__(self, config: TransformerSACConfig | None = None) -> None:
        super().__init__()
        self.config = config or TransformerSACConfig()
        self.actor = TransformerActor(self.config)
        self.critic1 = TransformerCritic(self.config)
        self.critic2 = TransformerCritic(self.config)
        self.actor_optim = torch.optim.AdamW(self.actor.parameters(), lr=self.config.actor_lr)
        self.critic_optim = torch.optim.AdamW(
            list(self.critic1.parameters()) + list(self.critic2.parameters()),
            lr=self.config.critic_lr,
        )

    def sample_action(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.actor.sample(obs)

    def update(self, obs: torch.Tensor, action: torch.Tensor, reward: torch.Tensor) -> dict[str, float]:
        q1 = self.critic1(obs, action)
        q2 = self.critic2(obs, action)
        target = reward.detach()
        critic_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)
        self.critic_optim.zero_grad(set_to_none=True)
        critic_loss.backward()
        nn.utils.clip_grad_norm_(list(self.critic1.parameters()) + list(self.critic2.parameters()), 1.0)
        self.critic_optim.step()

        sampled_action, log_prob, _ = self.actor.sample(obs)
        q_pi = torch.minimum(self.critic1(obs, sampled_action), self.critic2(obs, sampled_action))
        actor_loss = (self.config.alpha * log_prob - q_pi).mean()
        self.actor_optim.zero_grad(set_to_none=True)
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optim.step()

        return {
            "critic_loss": float(critic_loss.detach().cpu()),
            "actor_loss": float(actor_loss.detach().cpu()),
            "q_mean": float(q_pi.mean().detach().cpu()),
            "log_prob_mean": float(log_prob.mean().detach().cpu()),
        }

    def supervised_update(self, obs: torch.Tensor, target_action: torch.Tensor, sample_weight: torch.Tensor | None = None) -> dict[str, float]:
        mu, _ = self.actor(obs)
        pred_action = self.config.max_delta * torch.tanh(mu)
        loss_per_step = (pred_action - target_action.detach()) ** 2
        if sample_weight is not None:
            weight = sample_weight.detach().clamp(min=0.0).view(-1, 1)
            loss = (loss_per_step.mean(dim=1) * weight).sum() / torch.clamp(weight.sum(), min=1e-6)
        else:
            loss = loss_per_step.mean()
        self.actor_optim.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optim.step()
        return {
            "supervised_loss": float(loss.detach().cpu()),
            "target_delta_mean": float(target_action.mean().detach().cpu()),
            "target_delta_abs": float(target_action.abs().mean().detach().cpu()),
        }


class TransformerActor(nn.Module):
    def __init__(self, config: TransformerSACConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = SequenceEncoder(config.obs_dim, config.d_model, config.nhead, config.num_layers, config.dim_feedforward, config.dropout)
        self.mu = nn.Linear(config.d_model, 1)
        self.log_std = nn.Linear(config.d_model, 1)
        nn.init.zeros_(self.mu.weight)
        nn.init.zeros_(self.mu.bias)
        nn.init.zeros_(self.log_std.weight)
        nn.init.constant_(self.log_std.bias, -2.5)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(obs)
        mu = self.mu(h).squeeze(-1)
        log_std = torch.clamp(self.log_std(h).squeeze(-1), self.config.log_std_min, self.config.log_std_max)
        return mu, log_std

    def sample(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, log_std = self.forward(obs)
        std = log_std.exp()
        eps = torch.randn_like(mu)
        pre_tanh = mu + std * eps
        squashed = torch.tanh(pre_tanh)
        action = self.config.max_delta * squashed
        log_prob = -0.5 * (((pre_tanh - mu) / torch.clamp(std, min=1e-6)) ** 2 + 2.0 * log_std + math.log(2.0 * math.pi))
        log_prob = log_prob - torch.log(torch.clamp(1.0 - squashed**2, min=1e-6))
        return action, log_prob.sum(dim=1), mu


class TransformerCritic(nn.Module):
    def __init__(self, config: TransformerSACConfig) -> None:
        super().__init__()
        self.encoder = SequenceEncoder(config.obs_dim + 1, config.d_model, config.nhead, config.num_layers, config.dim_feedforward, config.dropout)
        self.head = nn.Sequential(
            nn.LayerNorm(config.d_model),
            nn.Linear(config.d_model, config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, 1),
        )

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([obs, action.unsqueeze(-1)], dim=-1)
        h = self.encoder(x).mean(dim=1)
        return self.head(h).squeeze(-1)


class SequenceEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(input_dim, d_model), nn.LayerNorm(d_model), nn.GELU())
        self.pos = SinusoidalPositionalEncoding(d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(self.pos(self.proj(x)))


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
        return x + self.pe[:, : x.shape[1]]
