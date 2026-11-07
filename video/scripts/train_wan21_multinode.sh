#!/usr/bin/env bash
# Perc-EC / Perc-VC on Wan2.1-T2V-1.3B across N nodes (the paper's runs used 12 x 8 = 96 GPUs).
#
# Brings up a Ray cluster over NODES, waits for every GPU to register, then runs the Hydra
# driver on the head. Ray places the ranks; there is no torchrun and no rank argument.
#
#   NODES="10.0.0.1 10.0.0.2 ... 10.0.0.12" \
#   PRETRAINED_MODEL=/path/to/Wan2.1-T2V-1.3B-Diffusers \
#     bash scripts/train_wan21_multinode.sh +pec.emb_weight=0.05 +pec.latent_weight=0.0
#
# NODES          space-separated IPs, head first. Must be ssh-reachable without a password.
# GPUS_PER_NODE  default 8.
# NUM_DEVICES    default NUM_NODES * GPUS_PER_NODE. Set it lower to leave GPUs free.
# DEGRADED_NODE  one IP that contributes GPUS_PER_NODE-1 GPUs (a dead card on that host).
#                The paper's 96-GPU runs used 95 this way.
# EXCLUDE_NODES  comma-separated IPs to drop entirely.
# SAVE_DIR, TRAIN_PROMPTS, EVAL_PROMPTS, NUM_ROLLOUTS, BATCH_SIZE, CONFIG   as in train_wan21.sh.
# ENV_FILE       optional file sourced on every node before Ray starts, for site-specific
#                settings (conda activation, proxy, HF_HOME, interface names). See ENV_FILE
#                below for the variables that matter.
#
# Everything after the options is passed through to Hydra.
set -uo pipefail

: "${NODES:?set NODES to the node IPs, head first}"
: "${PRETRAINED_MODEL:?set PRETRAINED_MODEL to the Wan2.1-T2V-1.3B-Diffusers directory}"

read -ra ALL_NODES <<< "$NODES"
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
RAY_PORT=${RAY_PORT:-6379}
CONFIG=${CONFIG:-diffusion/wan21/wan21_t2v_percec}
REPO=$(cd "$(dirname "$0")/.." && pwd)
LOGDIR=${LOGDIR:-$REPO/logs/multinode}
mkdir -p "$LOGDIR"

# Drop excluded hosts.
IFS=',' read -ra EXCLUDED <<< "${EXCLUDE_NODES:-}"
SELECTED=()
for n in "${ALL_NODES[@]}"; do
    skip=0
    for e in "${EXCLUDED[@]}"; do [ "$n" = "$e" ] && skip=1; done
    [ "$skip" -eq 0 ] && SELECTED+=("$n")
done
HEAD_IP=${SELECTED[0]}
WORKERS=("${SELECTED[@]:1}")
NUM_NODES=${#SELECTED[@]}

# A degraded host gives up one GPU, so the expected total is not simply nodes x gpus.
EXPECTED=$(( NUM_NODES * GPUS_PER_NODE ))
for n in "${SELECTED[@]}"; do
    [ "$n" = "${DEGRADED_NODE:-}" ] && EXPECTED=$(( EXPECTED - 1 ))
done
NUM_DEVICES=${NUM_DEVICES:-$EXPECTED}

# Per-node environment. Site-specific bits belong in ENV_FILE, not here.
#   The NCCL/IB settings below are what a 96-GPU run needed on an InfiniBand fabric; the
#   interface and HCA names are fabric-specific, so override them in ENV_FILE.
read -r -d '' COMMON_ENV <<ENV || true
export PYTHONPATH=$REPO:\${PYTHONPATH:-}
export PRETRAINED_MODEL=$PRETRAINED_MODEL
export TRAIN_PROMPTS=${TRAIN_PROMPTS:-$REPO/data/video_prompts.txt}
export EVAL_PROMPTS=${EVAL_PROMPTS:-$REPO/data/prompts100.txt}
export SAVE_DIR=${SAVE_DIR:-$REPO/saves/wan21_percec}
export HF_HOME=\${HF_HOME:-\$HOME/.cache/huggingface}
export HF_HUB_OFFLINE=\${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=\${TRANSFORMERS_OFFLINE:-1}
export NCCL_IB_DISABLE=\${NCCL_IB_DISABLE:-0}
export NCCL_IB_GID_INDEX=\${NCCL_IB_GID_INDEX:-3}
export NCCL_IB_TIMEOUT=\${NCCL_IB_TIMEOUT:-22}
export NCCL_IB_RETRY_CNT=\${NCCL_IB_RETRY_CNT:-7}
export NCCL_IB_QPS_PER_CONNECTION=\${NCCL_IB_QPS_PER_CONNECTION:-4}
export NCCL_CROSS_NIC=\${NCCL_CROSS_NIC:-0}
export NCCL_DEBUG=\${NCCL_DEBUG:-WARN}
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS=\${OMP_NUM_THREADS:-8}
export TOKENIZERS_PARALLELISM=false
# Writing .pyc onto a shared network filesystem hung workers in D state on page locks.
export PYTHONDONTWRITEBYTECODE=1
# One ffmpeg process per sample during video decode; 1024 fds silently produced empty mp4s.
ulimit -n 65536 2>/dev/null || true
# expandable_segments is deliberately off: it triggered an illegal memory access inside the
# FSDP2 split_with_sizes_copy, and measured fragmentation was only ~600 MB. Set ALLOC_CONF
# to turn it on for an A/B.
\${ALLOC_CONF:+export PYTORCH_CUDA_ALLOC_CONF=\$ALLOC_CONF}
${ENV_FILE:+source $ENV_FILE}
ENV

echo "=== Wan2.1 Perc-EC: ${NUM_NODES} nodes x ${GPUS_PER_NODE} GPUs -> ${NUM_DEVICES} devices ==="
echo "config=$CONFIG  head=$HEAD_IP  workers=${#WORKERS[@]}"

# 1. Clear any previous Ray cluster. This touches Ray only.
eval "$COMMON_ENV"; ray stop >/dev/null 2>&1
for ip in "${WORKERS[@]}"; do
    ssh -n -o ConnectTimeout=10 "$ip" "$COMMON_ENV
ray stop >/dev/null 2>&1" >/dev/null 2>&1 &
done
wait
echo "[1/4] previous Ray cluster cleared"

# 2. Head.
eval "$COMMON_ENV"
HEAD_GPUS=$GPUS_PER_NODE
[ "$HEAD_IP" = "${DEGRADED_NODE:-}" ] && HEAD_GPUS=$(( GPUS_PER_NODE - 1 ))
ray start --head --node-ip-address="$HEAD_IP" --port="$RAY_PORT" \
          --dashboard-host=0.0.0.0 --num-gpus="$HEAD_GPUS" \
          --disable-usage-stats > "$LOGDIR/head_raystart.log" 2>&1
echo "[2/4] head Ray started with $HEAD_GPUS GPUs"

# 3. Workers.
n=0
for ip in "${WORKERS[@]}"; do
    n=$((n + 1))
    ssh -n -o ConnectTimeout=10 "$ip" "$COMMON_ENV
cd $REPO
NG=$GPUS_PER_NODE
if [ '$ip' = '${DEGRADED_NODE:-none}' ]; then
    NG=\$(( $GPUS_PER_NODE - 1 ))
    export CUDA_VISIBLE_DEVICES=\$(seq -s, 0 \$(( NG - 1 )))
fi
ray stop >/dev/null 2>&1
ray start --address='$HEAD_IP:$RAY_PORT' --node-ip-address='$ip' --num-gpus=\$NG" \
        > "$LOGDIR/worker${n}.log" 2>&1 &
done
wait
echo "[3/4] ${#WORKERS[@]} workers asked to join"

# 4. Wait for the cluster to register every GPU, then run the driver on the head.
for _ in $(seq 1 30); do
    GOT=$(ray status 2>/dev/null | grep -oE '[0-9]+\.[0-9]+/[0-9]+\.[0-9]+ GPU' | head -1 |
          sed 's#.*/##;s# GPU##' | cut -d. -f1)
    [ "${GOT:-0}" -ge "$NUM_DEVICES" ] && break
    sleep 5
done
echo "[4/4] Ray GPUs: ${GOT:-?}/${NUM_DEVICES}"
if [ "${GOT:-0}" -lt "$NUM_DEVICES" ]; then
    echo "ERROR: only ${GOT:-0} of $NUM_DEVICES GPUs registered; check $LOGDIR/worker*.log" >&2
    exit 1
fi

cd "$REPO"
# return_embeddings is mandatory whenever emb_weight != 0: the reward backend has to ship
# per-sample PickScore embeddings, and the trainer raises if it gets none.
exec python -m unirl.train_diffusion \
    --config-name="$CONFIG" \
    num_devices="$NUM_DEVICES" \
    batch_size="${BATCH_SIZE:-$NUM_DEVICES}" \
    num_rollouts="${NUM_ROLLOUTS:-220}" \
    +devices_per_node=1 \
    backend.fsdp_cfg.activation_checkpointing=true \
    adv_use_global_std=false \
    +reward.backend.config.return_embeddings=true \
    "$@"
