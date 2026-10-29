#!/usr/bin/env bash
set -euo pipefail
repo=$1; root=$2
python="/home/data/$USER/envs/low-bit-vllm/bin/python"
export PYTHONPATH="$repo" HF_HOME="/home/data/$USER/hf-cache" WANDB_MODE=offline PYTHONUNBUFFERED=1
set -a; . "$HOME/.s3-cod.env"; set +a
"$python" "$repo/scripts/prepare_shared_manifest.py" --root "$root/audit"
"$python" "$repo/scripts/audit_shared_queue.py" --manifest "$root/audit/shared-manifest.json" \
    --prefix shveikinda/h200/low-bit/1029-distill-shared --output "$root/audit-summary.json" --wait-seconds 27000
