"""Built-in reward scorer implementations.

Trimmed to what the Wan2.1 Perc-EC / Perc-VC recipe needs: the frame-averaged
PickScore video reward and the image PickScore it wraps. Upstream UniRL also
ships aesthetic / CLIP / HPSv2-3 / ImageReward / GenEval2 / OCR / CLAP /
ImageBind / VideoAlign scorers; add them back from the upstream repo together
with their entries in ``registry.py`` if you need them.
"""

from .base import LocalRewardBackend
from .pickscore import PickScoreRewardScorer
from .registry import available_builtin_reward_models, resolve_builtin_reward_scorer_class
from .video import VideoRewardScorer

__all__ = [
    "LocalRewardBackend",
    "PickScoreRewardScorer",
    "VideoRewardScorer",
    "available_builtin_reward_models",
    "resolve_builtin_reward_scorer_class",
]
