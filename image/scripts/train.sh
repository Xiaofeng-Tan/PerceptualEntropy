#!/usr/bin/env bash
# Perc-EC / Perc-VC on FLUX.1-dev. One entry point for every recipe in config/grpo.py.
#
#   bash scripts/train.sh pickscore_flux_8gpu_reward_entropy_only_005   # Perc-EC, lambda=0.05
#   bash scripts/train.sh pickscore_flux_8gpu_pcvae_01                  # Perc-VC, w1=w2=0.1
#   bash scripts/train.sh pickscore_flux_8gpu_entropy_reg_005           # policy-entropy baseline
#   bash scripts/train.sh pickscore_flux_8gpu                           # Flow-GRPO baseline
#   bash scripts/train.sh pickscore_flux_8gpu_ddpo                      # DDPO baseline
#   bash scripts/train.sh pickscore_flux_8gpu_ddpo_pec_003              # Perc-EC on top of DDPO
#
#   NUM_GPUS=16 PORT=29600 bash scripts/train.sh <config>
#
# Checkpoints go to the recipe's config.save_dir (logs/<name> by default).
set -euo pipefail

CONFIG=${1:?usage: train.sh <config-name-in-config/grpo.py> [extra accelerate args]}
NUM_GPUS=${NUM_GPUS:-8}
PORT=${PORT:-29500}
cd "$(dirname "$0")/.."

# DeepSpeed ZeRO-2 is what the paper's runs used. If your accelerate version raises
# "no_sync context manager is incompatible with gradient partitioning logic of ZeRO
# stage 2", switch to scripts/accelerate_configs/multi_gpu.yaml -- plain DDP, same math.
ACCEL=${ACCEL:-scripts/accelerate_configs/deepspeed_zero2.yaml}

exec accelerate launch \
    --config_file "$ACCEL" \
    --num_processes="$NUM_GPUS" \
    --main_process_port "$PORT" \
    scripts/train_flux.py \
    --config "config/grpo.py:$CONFIG" "${@:2}"
