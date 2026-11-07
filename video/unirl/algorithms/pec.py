"""Perceptual Entropy Constraint (PEC) and PCVAE reward shaping for diffusion GRPO.

Port of pec-grpo (Tan et al., "Diversity Collapse despite Constant Entropy"),
``scripts/train_flux.py``: with ``dual_entropy_target="reward"`` and no importance
weighting, the reward is shaped *before* the GRPO advantage is taken::

    r' = r + emb_weight * H(reward_emb) - latent_weight * H(latent) + native_weight * H_policy

``H`` is the per-sample perceptual entropy of :func:`group_gaussian_entropy` —
``-log p`` under a diagonal Gaussian fitted to the sample's own prompt group
(``calculate_reward_emb_log_prob``). PEC is ``latent_weight = 0``; PCVAE sets both.

``native_weight`` is the plain GRPO entropy bonus, ``+ native_weight * H_policy`` with
``H_policy = -log pi(x)`` read off the SDE transition log-probs the rollout already stores —
the policy's own entropy in the generation space, not a fitted proxy. It is the paper's
native-entropy baseline: the SDE kernel's std comes from the sigma schedule and eta alone,
so E[-log pi] is a constant the model cannot move, which is the "entropy stays constant"
half of the title. PEC exists because that term cannot stop perceptual collapse.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch

# Component-reward key the reward backend uses to ship per-sample reward-model
# embeddings to the driver. Popped there before any logging or training sees it.
PEC_EMB_KEY = "_pec_emb"


@dataclass
class PECConfig:
    emb_weight: float = 0.0
    latent_weight: float = 0.0
    native_weight: float = 0.0

    @property
    def enabled(self) -> bool:
        return self.emb_weight != 0.0 or self.latent_weight != 0.0 or self.native_weight != 0.0

    @property
    def needs_latents(self) -> bool:
        return self.latent_weight != 0.0


def group_gaussian_entropy(x: torch.Tensor, n_groups: int) -> torch.Tensor:
    """Per-sample ``-mean_d log N(x_d; mu_g, var_g)`` with ``mu/var`` from the sample's group.

    ``x`` is ``(N, ...)`` with groups contiguous and equal-sized (the GRPO fork layout);
    returns ``(N,)``. Matches pec-grpo: population variance + 1e-8, dims averaged.
    """
    flat = x.detach().to(torch.float64).reshape(n_groups, x.shape[0] // n_groups, -1)
    mu = flat.mean(dim=1, keepdim=True)
    var = flat.var(dim=1, unbiased=False, keepdim=True) + 1e-8
    log_p = -0.5 * (flat - mu) ** 2 / var - 0.5 * math.log(2 * math.pi) - 0.5 * torch.log(var)
    return (-log_p.mean(dim=2)).reshape(-1).to(torch.float32)


def group_cosine_distance(emb: torch.Tensor, n_groups: int) -> float:
    """Mean pairwise cosine distance inside each group — a readable perceptual-diversity number."""
    e = torch.nn.functional.normalize(emb.detach().to(torch.float32).reshape(emb.shape[0], -1), dim=-1)
    e = e.reshape(n_groups, emb.shape[0] // n_groups, -1)
    k = e.shape[1]
    if k < 2:
        return 0.0
    sim = e @ e.transpose(1, 2)
    off_diag = (sim.sum(dim=(1, 2)) - sim.diagonal(dim1=1, dim2=2).sum(dim=1)) / (k * (k - 1))
    return float((1.0 - off_diag).mean().item())


def group_mean_variance(x: torch.Tensor, n_groups: int) -> float:
    """Within-group population variance, averaged over dims then groups (PEC's group_var logs)."""
    flat = x.detach().to(torch.float32).reshape(n_groups, x.shape[0] // n_groups, -1)
    return float(flat.var(dim=1, unbiased=False).mean().item())


def pec_shaped_rewards(
    rewards: torch.Tensor,
    n_groups: int,
    cfg: PECConfig,
    *,
    emb: Optional[torch.Tensor],
    latents: Optional[torch.Tensor],
    sde_logp: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Return ``r'`` and telemetry. Missing inputs for an enabled term raise, never silently skip."""
    rewards = rewards.to(torch.float32)
    bonus = torch.zeros_like(rewards)
    metrics: Dict[str, float] = {}
    if emb is not None:
        h_emb = group_gaussian_entropy(emb, n_groups)
        bonus = bonus + cfg.emb_weight * h_emb
        metrics["pec_H_emb_mean"] = float(h_emb.mean())
        metrics["pec_H_emb_std"] = float(h_emb.std())
        metrics["group_emb_var"] = group_mean_variance(emb, n_groups)
        metrics["group_emb_cos_dist"] = group_cosine_distance(emb, n_groups)
    elif cfg.emb_weight != 0.0:
        raise RuntimeError("PEC emb_weight != 0 but the reward backend returned no embeddings")
    if latents is not None:
        h_lat = group_gaussian_entropy(latents, n_groups)
        bonus = bonus - cfg.latent_weight * h_lat
        metrics["pec_H_latent_mean"] = float(h_lat.mean())
        metrics["pec_H_latent_std"] = float(h_lat.std())
        metrics["group_latent_var"] = group_mean_variance(latents, n_groups)
    elif cfg.needs_latents:
        raise RuntimeError("PCVAE latent_weight != 0 but no final latents are available")
    if sde_logp is not None:
        # Plain GRPO entropy bonus: H = -log pi(x), averaged over the trajectory's SDE steps.
        h_pol = -sde_logp.detach().to(torch.float32).mean(dim=1)
        bonus = bonus + cfg.native_weight * h_pol
        metrics["pec_H_policy_mean"] = float(h_pol.mean())
        metrics["pec_H_policy_std"] = float(h_pol.std())
        metrics["group_policy_H_var"] = group_mean_variance(h_pol.unsqueeze(1), n_groups)
    elif cfg.native_weight != 0.0:
        raise RuntimeError("native_weight != 0 but the rollout stored no SDE log-probs")
    metrics["pec_bonus_mean"] = float(bonus.mean())
    metrics["pec_bonus_std"] = float(bonus.std())
    metrics["pec_reward_std"] = float(rewards.std())
    return rewards + bonus, metrics


__all__ = [
    "PEC_EMB_KEY",
    "PECConfig",
    "group_cosine_distance",
    "group_gaussian_entropy",
    "group_mean_variance",
    "pec_shaped_rewards",
]
