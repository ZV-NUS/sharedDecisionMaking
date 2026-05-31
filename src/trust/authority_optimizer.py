from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch


@dataclass(frozen=True)
class AuthorityObjectiveWeights:
    safety: float = 8.0
    efficiency: float = 1.0
    comfort: float = 0.25
    smooth: float = 0.50


@dataclass(frozen=True)
class BayesianCEMConfig:
    iterations: int = 8
    candidates: int = 48
    elite_fraction: float = 0.22
    init_std: float = 0.20
    min_std: float = 0.015
    seed: int = 2026


class SurrogateAssistedRuleOptimizer:
    """Lightweight offline optimizer for fuzzy authority rule parameters.

    The intended research method is MOBO-EHVI. This implementation keeps the
    repository dependency-light by using a surrogate-assisted cross-entropy
    search interface: candidates are evaluated by rollout objectives, elites
    update a Gaussian proposal, and the best parameter vector can be saved as
    the offline rule base. The objective function can later be replaced by a
    BoTorch qEHVI/NEHVI loop without changing Work3 module interfaces.
    """

    def __init__(self, config: BayesianCEMConfig | None = None) -> None:
        self.config = config or BayesianCEMConfig()

    def optimize(
        self,
        initial_vector: torch.Tensor,
        evaluate: Callable[[torch.Tensor], dict[str, float]],
        weights: AuthorityObjectiveWeights | None = None,
    ) -> dict:
        cfg = self.config
        weights = weights or AuthorityObjectiveWeights()
        generator = torch.Generator()
        generator.manual_seed(int(cfg.seed))
        mean = initial_vector.detach().float().clone()
        std = torch.full_like(mean, float(cfg.init_std))
        best_vector = mean.clone()
        best_score = float("inf")
        best_metrics: dict[str, float] = {}
        history = []
        elite_count = max(2, int(round(cfg.candidates * cfg.elite_fraction)))

        for iteration in range(1, cfg.iterations + 1):
            noise = torch.randn((cfg.candidates, mean.numel()), generator=generator, dtype=mean.dtype)
            candidates = mean.view(1, -1) + noise * std.view(1, -1)
            candidates = torch.clamp(candidates, -2.0, 2.0)
            candidates[:, -2] = torch.clamp(candidates[:, -2], 0.05, 0.80)
            candidates[:, -1] = torch.clamp(candidates[:, -1], 0.0, 1.0)

            scored = []
            for idx in range(candidates.shape[0]):
                metrics = evaluate(candidates[idx])
                score = _scalarize(metrics, weights)
                scored.append((score, candidates[idx].clone(), metrics))
                if score < best_score:
                    best_score = float(score)
                    best_vector = candidates[idx].clone()
                    best_metrics = dict(metrics)

            scored.sort(key=lambda item: item[0])
            elites = torch.stack([item[1] for item in scored[:elite_count]], dim=0)
            mean = elites.mean(dim=0)
            std = torch.maximum(elites.std(dim=0, unbiased=False), torch.full_like(std, float(cfg.min_std)))
            history.append(
                {
                    "iteration": iteration,
                    "best_score": float(scored[0][0]),
                    "global_best_score": float(best_score),
                    "best_metrics": scored[0][2],
                }
            )

        return {
            "best_vector": best_vector,
            "best_score": best_score,
            "best_metrics": best_metrics,
            "history": history,
        }


def _scalarize(metrics: dict[str, float], weights: AuthorityObjectiveWeights) -> float:
    return (
        weights.safety * float(metrics.get("safety", 0.0))
        + weights.efficiency * float(metrics.get("efficiency", 0.0))
        + weights.comfort * float(metrics.get("comfort", 0.0))
        + weights.smooth * float(metrics.get("smooth", 0.0))
    )
