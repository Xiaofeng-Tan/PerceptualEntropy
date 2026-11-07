"""Helpers for building structured WandB metrics in train loops."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import torch


def _tensor_stats(prefix: str, tensor: Optional[torch.Tensor]) -> Dict[str, float]:
    if tensor is None or (not torch.is_tensor(tensor)) or tensor.numel() == 0:
        return {}
    flat = tensor.detach().to(dtype=torch.float32).reshape(-1).cpu()
    # Non-finite marks "not scored" rows (per-domain component rewards use NaN).
    flat = flat[torch.isfinite(flat)]
    if flat.numel() == 0:
        return {}
    return {
        f"{prefix}_mean": float(flat.mean().item()),
        f"{prefix}_std": float(flat.std(unbiased=False).item()),
        f"{prefix}_min": float(flat.min().item()),
        f"{prefix}_max": float(flat.max().item()),
    }


def _zero_std_group_counts_from_ids(
    rewards: torch.Tensor,
    group_ids: Optional[List[str]],
) -> tuple[int, int]:
    if not isinstance(group_ids, list) or len(group_ids) != int(rewards.shape[0]):
        return 0, 0
    ordered: Dict[str, List[float]] = {}
    rewards_f = rewards.to(dtype=torch.float32).reshape(-1)
    for sample_idx, raw_group_id in enumerate(group_ids):
        group_id = str(raw_group_id).strip()
        if not group_id:
            continue
        ordered.setdefault(group_id, []).append(float(rewards_f[sample_idx].item()))
    if not ordered:
        return 0, 0
    zero_std = 0
    for values in ordered.values():
        if len(values) <= 1:
            continue
        std = torch.tensor(values, dtype=torch.float32).std(unbiased=False)
        if float(std.item()) <= 1e-8:
            zero_std += 1
    return zero_std, len(ordered)


def group_variance_from_ids(
    values: torch.Tensor,
    group_ids: Optional[List[str]],
) -> Optional[float]:
    """Mean within-group variance, as in DanceGRPO's ``pec_group_logs`` and PEC's
    ``calculate_group_variance``: bucket rows by prompt group, take the per-dimension
    population variance inside each group, average over dimensions, then over groups.

    ``values`` is ``(N, ...)``; everything past the batch dim is flattened. Groups with
    fewer than two members carry no variance and are skipped.
    """
    if not isinstance(group_ids, list) or int(values.shape[0]) != len(group_ids):
        return None
    flat = values.detach().to(dtype=torch.float32).reshape(int(values.shape[0]), -1).cpu()
    buckets: Dict[str, List[int]] = {}
    for sample_idx, raw_group_id in enumerate(group_ids):
        group_id = str(raw_group_id).strip()
        if group_id:
            buckets.setdefault(group_id, []).append(sample_idx)
    per_group: List[float] = []
    for indices in buckets.values():
        if len(indices) < 2:
            continue
        rows = flat[torch.tensor(indices, dtype=torch.long)]
        rows = rows[torch.isfinite(rows).all(dim=1)] if rows.dim() == 2 else rows
        if rows.shape[0] < 2:
            continue
        per_group.append(float(rows.var(dim=0, unbiased=False).mean().item()))
    if not per_group:
        return None
    return sum(per_group) / len(per_group)


def vae_space_group_variance(part: Any, *, pool_hw: int = 32) -> Optional[float]:
    """Within-group variance of the decoded videos — PEC's ``group_latent_var`` term.

    PEC reads the raw final latents; UniRL keeps only the VAE *output* on the part, so
    the variance is measured there. Frames are average-pooled to ``pool_hw`` first:
    the full tensors are GB-scale per rollout and the pooled statistic tracks the same
    collapse (identical generations -> variance 0).
    """
    primitives = getattr(part, "primitives", None)
    decoded = primitives.get("video") if primitives else None
    if decoded is None:
        return None
    try:
        from unirl.distributed.tensor.ref import hydrate

        per_sample = hydrate(decoded).to_list()
    except Exception:
        return None
    pooled: List[torch.Tensor] = []
    for video in per_sample:
        frames = getattr(video, "frames", None)
        if not torch.is_tensor(frames) or frames.dim() != 4:
            return None
        x = frames.detach().to(dtype=torch.float32).cpu()
        pooled.append(torch.nn.functional.adaptive_avg_pool2d(x, (pool_hw, pool_hw)).reshape(-1))
    if len(pooled) < 2:
        return None
    sizes = {int(p.numel()) for p in pooled}
    if len(sizes) != 1:
        return None
    return group_variance_from_ids(torch.stack(pooled), getattr(part, "group_ids", None))


def compute_rollout_sample_metrics(*, sample: Any, trunc_len: Optional[int] = None) -> Dict[str, float]:
    """Build rollout metrics directly from a :class:`Sample`."""
    from unirl.types.sampling import ARSamplingParams

    metrics: Dict[str, float] = {}
    if sample is None:
        metrics["num_samples"] = 0
        return metrics

    gen_parts = sample.gen_parts()
    metrics["num_samples"] = gen_parts[-1].batch_size if gen_parts else sample.batch_size
    multi = len(gen_parts) > 1
    for part in gen_parts:
        name = "ar" if isinstance(part.sampling_params, ARSamplingParams) else "diffusion"
        prefix = f"{name}_" if multi else ""
        rewards = part.rewards
        if torch.is_tensor(rewards) and rewards.numel() > 0:
            rewards_f = rewards.detach().to(dtype=torch.float32).reshape(-1).cpu()
            metrics.update(_tensor_stats(f"{prefix}reward", rewards_f))
            zero_cnt, group_cnt = _zero_std_group_counts_from_ids(
                rewards_f,
                part.group_ids,
            )
            if group_cnt > 0:
                metrics[f"{prefix}zero_std_group_ratio"] = float(zero_cnt) / float(group_cnt)
                metrics[f"{prefix}zero_std_group_count"] = float(zero_cnt)
                metrics[f"{prefix}group_count"] = float(group_cnt)
            # PEC group-collapse telemetry: spread of a group in reward space and in
            # VAE-output space. Both go to zero when a group's samples stop differing,
            # which is what drives the advantages (and so the gradient) to zero.
            reward_var = group_variance_from_ids(rewards_f.reshape(-1, 1), part.group_ids)
            if reward_var is not None:
                metrics[f"{prefix}group_reward_var"] = reward_var
            vae_var = vae_space_group_variance(part)
            if vae_var is not None:
                metrics[f"{prefix}group_vae_var"] = vae_var

        advantages = part.advantages
        if torch.is_tensor(advantages) and advantages.numel() > 0:
            adv_f = advantages.detach().to(dtype=torch.float32).reshape(-1).cpu()
            metrics.update(_tensor_stats(f"{prefix}advantage", adv_f))

        segment = part.segment
        lengths = segment.lengths if segment is not None else None
        if torch.is_tensor(lengths) and lengths.numel() > 0:
            len_f = lengths.detach().to(dtype=torch.float32).reshape(-1).cpu()
            metrics.update(_tensor_stats(f"{prefix}response_len", len_f))
            if trunc_len is not None and int(trunc_len) > 0:
                metrics[f"{prefix}trunc_ratio"] = float(
                    (len_f >= float(int(trunc_len))).to(dtype=torch.float32).mean().item()
                )

        component_rewards = part.component_rewards
        if isinstance(component_rewards, dict):
            for cname, tensor in component_rewards.items():
                if not torch.is_tensor(tensor) or tensor.numel() == 0:
                    continue
                safe_name = str(cname).replace("/", "_")
                cat = tensor.detach().to(dtype=torch.float32).reshape(-1).cpu()
                metrics.update(_tensor_stats(f"{prefix}reward_{safe_name}", cat))

    return metrics


def pooled_window_reward_metrics(parts: Sequence[Any]) -> Dict[str, float]:
    """Reward metrics pooled over an accumulation window's gen Parts."""
    metrics: Dict[str, float] = {"num_samples": sum(p.batch_size for p in parts)}
    rewards = [p.rewards for p in parts]
    if all(torch.is_tensor(r) and r.numel() > 0 for r in rewards):
        pooled = torch.cat([r.detach().to(dtype=torch.float32).reshape(-1).cpu() for r in rewards])
        metrics.update(_tensor_stats("reward", pooled))
        group_ids = [g for p in parts for g in p.group_ids]
        zero_cnt, group_cnt = _zero_std_group_counts_from_ids(pooled, group_ids)
        if group_cnt > 0:
            metrics["zero_std_group_ratio"] = float(zero_cnt) / float(group_cnt)
            metrics["zero_std_group_count"] = float(zero_cnt)
            metrics["group_count"] = float(group_cnt)
    components = [p.component_rewards for p in parts]
    if components and all(isinstance(c, dict) for c in components):
        for cname in sorted(set.intersection(*(set(c) for c in components))):
            tensors = [c[cname] for c in components]
            if all(torch.is_tensor(t) and t.numel() > 0 for t in tensors):
                pooled_c = torch.cat([t.detach().to(dtype=torch.float32).reshape(-1).cpu() for t in tensors])
                metrics.update(_tensor_stats(f"reward_{str(cname).replace('/', '_')}", pooled_c))
    return metrics
