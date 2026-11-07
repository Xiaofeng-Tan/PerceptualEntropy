"""Built-in reward scorer / spec registry."""

from __future__ import annotations

import importlib
from typing import Dict, Tuple, Type

from unirl.reward.base import BaseRewardComponentSpec, RewardBackend

_BUILTIN_REGISTRY: Dict[str, Tuple[str, str, str]] = {
    "pickscore": ("unirl.reward.local.pickscore", "PickScoreRewardScorer", "PickScoreSpec"),
}


def available_builtin_reward_models() -> Tuple[str, ...]:
    """Return built-in reward model names."""
    return tuple(sorted(_BUILTIN_REGISTRY.keys()))


def _resolve_builtin_entry(model_name: str) -> Tuple[str, str, str]:
    key = "" if model_name is None else str(model_name).strip().lower()
    if key not in _BUILTIN_REGISTRY:
        raise ValueError(f"Unknown model_name: {model_name}. Available: {list(available_builtin_reward_models())}")
    return _BUILTIN_REGISTRY[key]


def resolve_builtin_reward_scorer_class(model_name: str) -> Type[RewardBackend]:
    """Resolve a built-in reward scorer class by canonical model name."""
    module_name, scorer_attr, _ = _resolve_builtin_entry(model_name)
    module = importlib.import_module(module_name)
    scorer_cls = getattr(module, scorer_attr)
    if not isinstance(scorer_cls, type) or not issubclass(scorer_cls, RewardBackend):
        raise TypeError(f"Configured scorer {module_name}.{scorer_attr} is not a RewardBackend.")
    return scorer_cls


def resolve_builtin_reward_spec_class(model_name: str) -> Type[BaseRewardComponentSpec]:
    """Resolve a built-in reward Spec class by canonical model name."""
    module_name, _, spec_attr = _resolve_builtin_entry(model_name)
    module = importlib.import_module(module_name)
    spec_cls = getattr(module, spec_attr)
    if not isinstance(spec_cls, type) or not issubclass(spec_cls, BaseRewardComponentSpec):
        raise TypeError(f"Configured spec {module_name}.{spec_attr} is not a BaseRewardComponentSpec.")
    return spec_cls


__all__ = [
    "available_builtin_reward_models",
    "resolve_builtin_reward_scorer_class",
    "resolve_builtin_reward_spec_class",
]
