from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass(frozen=True)
class IT2TSKAuthorityConfig:
    centers: tuple[float, ...] = (0.15, 0.50, 0.85)
    sigmas: tuple[float, ...] = (0.18, 0.22, 0.18)
    uncertainty_scale: float = 0.35
    conservative_kappa: float = 0.55
    smooth_beta: float = 0.55
    context_gain: float = 1.2
    context_smooth_boost: float = 0.30
    min_authority: float = 0.0
    max_authority: float = 1.0
    consequent_p: tuple[float, ...] = field(default_factory=tuple)
    consequent_q: tuple[float, ...] = field(default_factory=tuple)
    consequent_s: tuple[float, ...] = field(default_factory=tuple)


class IntervalType2TSKAuthority:
    """Interval Type-2 TSK fuzzy reference authority inference.

    Input at each prediction step is two-way trust:
    x = T(machine -> human), y = T(human -> machine).
    Output is human reference authority lambda_h in [0, 1].
    The same rule base is shared across all prediction steps.
    """

    def __init__(self, config: IT2TSKAuthorityConfig | None = None) -> None:
        self.config = config or IT2TSKAuthorityConfig()
        self.num_terms = len(self.config.centers)
        self.num_rules = self.num_terms * self.num_terms
        self._p, self._q, self._s = self._build_consequents()

    def infer(
        self,
        trust_machine_to_human: torch.Tensor,
        trust_human_to_machine: torch.Tensor,
        environment_urgency: torch.Tensor | None = None,
        initial_authority: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        x = torch.clamp(trust_machine_to_human, 0.0, 1.0)
        y = torch.clamp(trust_human_to_machine, 0.0, 1.0)
        lower_x, upper_x = self._it2_membership(x)
        lower_y, upper_y = self._it2_membership(y)
        p = self._p.to(device=x.device, dtype=x.dtype)
        q = self._q.to(device=x.device, dtype=x.dtype)
        s = self._s.to(device=x.device, dtype=x.dtype)

        lower_num = torch.zeros_like(x)
        lower_den = torch.zeros_like(x)
        upper_num = torch.zeros_like(x)
        upper_den = torch.zeros_like(x)
        rule_outputs = []
        idx = 0
        for i in range(self.num_terms):
            for j in range(self.num_terms):
                lower_fire = lower_x[..., i] * lower_y[..., j]
                upper_fire = upper_x[..., i] * upper_y[..., j]
                out = torch.clamp(p[idx] * x + q[idx] * y + s[idx], self.config.min_authority, self.config.max_authority)
                lower_num = lower_num + lower_fire * out
                lower_den = lower_den + lower_fire
                upper_num = upper_num + upper_fire * out
                upper_den = upper_den + upper_fire
                rule_outputs.append(out)
                idx += 1

        lower = lower_num / torch.clamp(lower_den, min=1e-6)
        upper = upper_num / torch.clamp(upper_den, min=1e-6)
        lo = torch.minimum(lower, upper)
        hi = torch.maximum(lower, upper)
        raw = self.config.conservative_kappa * lo + (1.0 - self.config.conservative_kappa) * hi
        if environment_urgency is not None:
            urgency = torch.clamp(environment_urgency.to(device=raw.device, dtype=raw.dtype), 0.0, 1.0)
            raw = torch.clamp(0.5 + (1.0 + self.config.context_gain * urgency) * (raw - 0.5), self.config.min_authority, self.config.max_authority)
        smooth = self._smooth_authority(raw, initial_authority, environment_urgency)
        return {
            "authority_ref": smooth,
            "authority_raw": raw,
            "authority_lower": lo,
            "authority_upper": hi,
            "rule_outputs": torch.stack(rule_outputs, dim=-1),
        }

    def with_parameters(self, vector: torch.Tensor | list[float]) -> "IntervalType2TSKAuthority":
        values = torch.as_tensor(vector, dtype=torch.float32).flatten()
        n = self.num_rules
        if values.numel() != 3 * n + 2:
            raise ValueError(f"Expected {3 * n + 2} parameters, got {values.numel()}")
        cfg = IT2TSKAuthorityConfig(
            centers=self.config.centers,
            sigmas=self.config.sigmas,
            uncertainty_scale=float(torch.clamp(values[-2], 0.05, 0.80)),
            conservative_kappa=float(torch.clamp(values[-1], 0.0, 1.0)),
            smooth_beta=self.config.smooth_beta,
            min_authority=self.config.min_authority,
            max_authority=self.config.max_authority,
            context_gain=self.config.context_gain,
            context_smooth_boost=self.config.context_smooth_boost,
            consequent_p=tuple(float(v) for v in values[:n]),
            consequent_q=tuple(float(v) for v in values[n : 2 * n]),
            consequent_s=tuple(float(v) for v in values[2 * n : 3 * n]),
        )
        return IntervalType2TSKAuthority(cfg)

    def parameter_vector(self) -> torch.Tensor:
        return torch.cat(
            [
                self._p,
                self._q,
                self._s,
                torch.tensor([self.config.uncertainty_scale, self.config.conservative_kappa], dtype=torch.float32),
            ]
        )

    def _it2_membership(self, value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        centers = torch.as_tensor(self.config.centers, dtype=value.dtype, device=value.device).view(*([1] * value.ndim), -1)
        sigmas = torch.as_tensor(self.config.sigmas, dtype=value.dtype, device=value.device).view(*([1] * value.ndim), -1)
        lower_sigma = torch.clamp(sigmas * (1.0 - self.config.uncertainty_scale), min=0.03)
        upper_sigma = torch.clamp(sigmas * (1.0 + self.config.uncertainty_scale), min=0.03)
        expanded = value.unsqueeze(-1)
        lower = torch.exp(-0.5 * ((expanded - centers) / lower_sigma) ** 2)
        upper = torch.exp(-0.5 * ((expanded - centers) / upper_sigma) ** 2)
        return torch.minimum(lower, upper), torch.maximum(lower, upper)

    def _smooth_authority(
        self,
        raw: torch.Tensor,
        initial_authority: torch.Tensor | None,
        environment_urgency: torch.Tensor | None,
    ) -> torch.Tensor:
        smoothed = torch.zeros_like(raw)
        if initial_authority is None:
            previous = raw[:, 0] if raw.ndim == 2 else raw[..., 0]
        else:
            previous = initial_authority.to(device=raw.device, dtype=raw.dtype)
        for i in range(raw.shape[1]):
            beta = torch.full_like(previous, float(self.config.smooth_beta))
            if environment_urgency is not None:
                urgency_i = torch.clamp(environment_urgency[:, i].to(device=raw.device, dtype=raw.dtype), 0.0, 1.0)
                beta = torch.clamp(beta + float(self.config.context_smooth_boost) * urgency_i, max=0.95)
            previous = (1.0 - beta) * previous + beta * raw[:, i]
            smoothed[:, i] = previous
        return torch.clamp(smoothed, self.config.min_authority, self.config.max_authority)

    def _build_consequents(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.config.consequent_p and self.config.consequent_q and self.config.consequent_s:
            return (
                torch.tensor(self.config.consequent_p, dtype=torch.float32),
                torch.tensor(self.config.consequent_q, dtype=torch.float32),
                torch.tensor(self.config.consequent_s, dtype=torch.float32),
            )
        p = []
        q = []
        s = []
        centers = list(self.config.centers)
        for tmh in centers:
            for htm in centers:
                target = float(max(0.0, min(1.0, 0.5 + 0.45 * (tmh - htm))))
                p.append(0.35)
                q.append(-0.35)
                s.append(target - 0.35 * tmh + 0.35 * htm)
        return torch.tensor(p), torch.tensor(q), torch.tensor(s)
