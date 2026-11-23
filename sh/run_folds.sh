#!/usr/bin/env bash
# Three independent patient folds. A worker owns one GPU at a time.
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"
config="$1"
run="${RUN_NAME:-main}"
py="${PYTHON:-python}"
IFS=',' read -r -a gpus <<< "${GPUS:-0,1,2}"
if (( ${#gpus[@]} == 0 )); then
    printf '%s\n' 'Set GPUS to a comma-separated list of CUDA device indices.' >&2
    exit 2
fi
workers=$(( ${#gpus[@]} > 3 ? 3 : ${#gpus[@]} ))
pids=()
trap 'for pid in "${pids[@]}"; do kill "$pid" 2>/dev/null || true; done' INT TERM
for ((worker=0; worker<workers; worker++)); do
    (
        for ((fold=worker+1; fold<=3; fold+=workers)); do
            CUDA_VISIBLE_DEVICES="${gpus[worker]}" "$py" -B train.py \
                --config "$config" --fold "$fold" --device cuda:0 --run "$run"
        done
    ) &
    pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do
    wait "$pid" || failed=1
done
exit "$failed"
