"""unirl stage-driven algorithms; concrete algorithms resolve lazily (PEP 562) so ``base`` imports stay light."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS = (
    ("AlgorithmStepResult", "base"),
    ("FlowGRPO", "flowgrpo"),
    ("FlowGRPOConfig", "flowgrpo"),
)
_SYMBOL_MODULES = dict(_EXPORTS)


def __getattr__(name: str) -> Any:
    try:
        module = _SYMBOL_MODULES[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    return getattr(import_module(f".{module}", __name__), name)


def __dir__() -> list[str]:
    return sorted(__all__)


__all__ = [name for name, _ in _EXPORTS]
