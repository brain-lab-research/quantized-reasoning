#!/usr/bin/env bash
# Run in a separate tmux window. No GPU use while the current parent is alive.
set -euo pipefail
parent=$1
repo=$2
root=$3
while kill -0 "$parent" 2>/dev/null; do sleep 30; done
if ! grep -q '^JOB_EXIT 0$' "$root/run-continuous.log"; then
    printf 'PRIMARY_FAILED: preserving partial results and retrying once\n'
    bash "$repo/vv/job.sh" "$repo" "$root" s3 continuous >> "$root/run-continuous.log" 2>&1
fi
if [[ ! -f "$root/models-ready.txt" ]]; then
    printf 'MODELS_NOT_READY: inspect model-download.log\n'
    exit 1
fi
bash "$repo/vv/job.sh" "$repo" "$root" s3 family > "$root/run-family.log" 2>&1
