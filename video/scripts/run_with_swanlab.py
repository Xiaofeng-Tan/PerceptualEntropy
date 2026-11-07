"""Run unirl.train_diffusion with SwanLab capturing UniRL's metrics and media.

UniRL only implements a wandb backend, so two bridges are installed here and
nothing in UniRL's training path is touched:

1. Scalars: ``swanlab.sync_wandb()`` mirrors every ``wandb.log`` scalar into
   SwanLab. Metrics UniRL already emits (DanceGRPO parity):
     train/loss, train/grad_norm, train/lr   <- DanceGRPO loss/grad_norm/learning_rate
     perf/step_time_s, perf/<phase>_time_s   <- DanceGRPO step_time/avg_step_time
     rollout/reward{,_std,_min,_max}          <- DanceGRPO reward_* stats
     rollout/zero_std_group_ratio, group_count, eval/*

2. Media: sync_wandb carries scalars only (verified: its local media/ dir stays
   empty for wandb.Video payloads), so UniRLWandBLogger.log_generated_media is
   wrapped to encode each preview video to GIF and hand it to swanlab.Video.
   GIF because swanlab 0.10.2's Video accepts only ``.gif`` (0.10.2 is the
   newest release on the mirror, so mp4 is not available).
"""
import os
import runpy
import sys
import tempfile

import numpy as np
import swanlab

swanlab.sync_wandb(wandb_run=False)  # scalars -> SwanLab; no parallel cloud wandb run

# NB: do not read SWANLAB_PROJECT — swanlab's own pydantic settings source
# parses that env var and rejects a plain string. Use a private name.
project = os.environ.pop("SL_PROJECT", None) or "perceptual-entropy-video"
# SL_RESUME_ID appends to an existing run so a checkpoint resume continues the same curves;
# SL_BACKFILL (json [[step, key, value], ...], built by swanlab_backfill.py) is written first to
# fill the steps that ran in a separate run, and the auto step then continues past it.
resume_id = os.environ.get("SL_RESUME_ID")
swanlab.init(
    project=project,
    experiment_name=os.environ.get("SL_RUN", "wan21_t2v_1p3b_95gpu"),
    config={"note": "UniRL FlowGRPO Wan2.1-T2V-1.3B"},
    **({"id": resume_id, "resume": "must"} if resume_id else {}),
)
# swanlab prints its own project/run URLs to stdout at init; the launcher greps them.

if os.environ.get("SL_BACKFILL"):
    import json
    from collections import defaultdict

    rows = defaultdict(dict)
    for step, key, value in json.load(open(os.environ["SL_BACKFILL"])):
        rows[int(step)][key] = value
    for step in sorted(rows):
        swanlab.log(rows[step], step=step)
    print(f"[swanlab] backfilled {len(rows)} steps ({min(rows)}..{max(rows)})", flush=True)


def _to_gif(video, fps: int) -> str:
    """Encode one channel-first (C, T, H, W) video tensor to a GIF path."""
    import imageio
    import torch

    v = video
    if torch.is_tensor(v):
        if v.dim() == 5:
            v = v.squeeze(0)
        arr = v.detach().to("cpu", dtype=torch.float32).clamp(0, 1).mul(255).to(torch.uint8).numpy()
    else:
        arr = np.asarray(v)
    if arr.ndim == 4 and arr.shape[0] in (1, 3, 4):  # (C, T, H, W) -> (T, H, W, C)
        arr = arr.transpose(1, 2, 3, 0)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    path = tempfile.mktemp(suffix=".gif")
    imageio.mimwrite(path, arr, format="GIF", duration=1.0 / max(1, int(fps)), loop=0)
    return path


def _install_media_bridge() -> None:
    from unirl.utils.wandb_logger import UniRLWandBLogger

    original = UniRLWandBLogger.log_generated_media

    def log_generated_media(self, step, media_preview, *, video_fps: int = 8, **kwargs):
        result = original(self, step, media_preview, video_fps=video_fps, **kwargs)
        videos = getattr(media_preview, "videos", None) or []
        if not videos:
            return result
        key = kwargs.get("video_key") or (
            kwargs.get("key", "rollout/generated_media")
            .replace("_media", "_videos")
            .replace("/generated_media", "/generated_videos")
        )
        prompts = list(getattr(media_preview, "prompts", None) or [])
        rewards = list(getattr(media_preview, "rewards", None) or [])
        clips = []
        for idx, video in enumerate(videos):
            try:
                caption = prompts[idx][:100] if idx < len(prompts) else None
                if idx < len(rewards):
                    caption = f"{caption or ''} | reward: {float(rewards[idx]):.3f}"
                clips.append(swanlab.Video(_to_gif(video, video_fps), caption=caption))
            except Exception as exc:  # a preview must never break training
                print(f"[swanlab] skipped video {idx}: {type(exc).__name__}: {exc}", flush=True)
        if clips:
            try:
                swanlab.log({key: clips})
            except Exception as exc:
                print(f"[swanlab] media log failed: {type(exc).__name__}: {exc}", flush=True)
        return result

    UniRLWandBLogger.log_generated_media = log_generated_media
    print("[swanlab] media bridge installed (videos -> GIF -> swanlab.Video)", flush=True)


_install_media_bridge()

sys.argv[0] = "unirl.train_diffusion"
runpy.run_module("unirl.train_diffusion", run_name="__main__")
