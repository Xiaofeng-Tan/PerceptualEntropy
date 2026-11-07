#!/usr/bin/env bash
# Perc-EC / Perc-VC on Wan2.1-T2V-1.3B.
#
#   PRETRAINED_MODEL=/path/to/Wan2.1-T2V-1.3B-Diffusers bash scripts/train_wan21.sh percec 0.05 0.0
#   PRETRAINED_MODEL=...                                 bash scripts/train_wan21.sh percvc 0.05 0.05
#
# Args: <tag> <emb_weight> <latent_weight>
#   emb_weight    lambda on H_perc(reward embedding)  -> Perc-EC
#   latent_weight weight on -H(VAE latent)            -> Perc-VC when non-zero
#
# Single node by default. For N nodes, start a Ray cluster first (`ray start --head`
# on one node, `ray start --address=<head>:6379` on the rest) and set NUM_DEVICES to
# the total GPU count; Hydra/Ray then places the workers.
set -euo pipefail

TAG=${1:?usage: train_wan21.sh <tag> <emb_weight> <latent_weight>}
W1=${2:?}
W2=${3:?}

: "${PRETRAINED_MODEL:?set PRETRAINED_MODEL to the Wan2.1-T2V-1.3B-Diffusers directory}"
export PRETRAINED_MODEL
export TRAIN_PROMPTS=${TRAIN_PROMPTS:-data/video_prompts.txt}
export EVAL_PROMPTS=${EVAL_PROMPTS:-data/prompts100.txt}
export SAVE_DIR=${SAVE_DIR:-saves/wan21_$TAG}
# PickScore + CLIP-ViT-H are fetched from the HF cache; pre-download them to run offline.
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
export PYTHONDONTWRITEBYTECODE=1

NUM_DEVICES=${NUM_DEVICES:-8}
NUM_ROLLOUTS=${NUM_ROLLOUTS:-220}

# return_embeddings is mandatory whenever emb_weight != 0: the reward backend has to ship
# per-sample PickScore embeddings, and the trainer raises if it gets none.
exec python -m unirl.train_diffusion \
    --config-name=diffusion/wan21/wan21_t2v_percec \
    num_devices="$NUM_DEVICES" \
    batch_size="${BATCH_SIZE:-$NUM_DEVICES}" \
    num_rollouts="$NUM_ROLLOUTS" \
    backend.fsdp_cfg.activation_checkpointing=true \
    adv_use_global_std=false \
    +reward.backend.config.return_embeddings=true \
    +pec.emb_weight="$W1" \
    +pec.latent_weight="$W2" \
    "${@:4}"
