#!/usr/bin/env bash
# Sequential Perc-EC / Perc-VC sweep on Wan2.1-T2V, every setting trained to the same number of
# rollouts. Only the entropy weights differ between jobs.
#
#   NODES="10.0.0.1 ... 10.0.0.12" \
#   PRETRAINED_MODEL=/path/to/Wan2.1-T2V-1.3B-Diffusers \
#     bash scripts/sweep_wan21.sh
#
# Each job resumes from the newest checkpoint under its own save directory, so re-running the
# script after an interruption continues rather than restarting. A job whose final checkpoint
# already exists is skipped.
#
# STEPS            rollouts per job (default 220)
# SAVE_ROOT        where the per-job save directories live (default <repo>/saves)
# Everything else (NODES, DEGRADED_NODE, GPUS_PER_NODE, NUM_DEVICES, ENV_FILE, ...) is read by
# train_wan21_multinode.sh; see that script.
set -u

REPO=$(cd "$(dirname "$0")/.." && pwd)
STEPS=${STEPS:-220}
SAVE_ROOT=${SAVE_ROOT:-$REPO/saves}
LOGDIR=${LOGDIR:-$REPO/logs/sweep}
mkdir -p "$SAVE_ROOT" "$LOGDIR"

COMMON=(
    num_rollouts="$STEPS"
    logging.log_media=true
    logging.media_max_items=3
    logging.media_log_interval=1
)

# name      emb_w  latent_w  save_subdir
JOBS=(
    "percec003  0.03  0.0   wan21_percec003"
    "percec005  0.05  0.0   wan21_percec005"
    "percec01   0.1   0.0   wan21_percec01"
    "percvc003  0.03  0.03  wan21_percvc003"
    "percvc005  0.05  0.05  wan21_percvc005"
    "percvc01   0.1   0.1   wan21_percvc01"
)

for job in "${JOBS[@]}"; do
    read -r name w1 w2 save <<< "$job"
    dir=$SAVE_ROOT/$save
    if [ -f "$dir/checkpoint-$STEPS/trainer_state.json" ]; then
        echo "$(date '+%F %T') skip $name (checkpoint-$STEPS exists)"
        continue
    fi
    latest=$(ls "$dir" 2>/dev/null | grep -E '^checkpoint-[0-9]+$' | sort -V | tail -1)
    extra=()
    [ -n "$latest" ] && extra+=("+load_dir=$dir/$latest")
    echo "$(date '+%F %T') start $name  emb=$w1 latent=$w2  from ${latest:-scratch}"

    SAVE_DIR=$dir bash "$REPO/scripts/train_wan21_multinode.sh" \
        "${COMMON[@]}" \
        +pec.emb_weight="$w1" \
        +pec.latent_weight="$w2" \
        "${extra[@]}" \
        >> "$LOGDIR/$name.log" 2>&1 < /dev/null

    echo "$(date '+%F %T') end $name: $(grep -oE 'EVAL step [0-9]+' "$LOGDIR/$name.log" | tail -1)"
done
