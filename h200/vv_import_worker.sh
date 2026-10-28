#!/usr/bin/env bash
set -euo pipefail
repo=$1
root=$2
export PYTHONPATH="$repo" HF_HOME="/home/data/$USER/hf-cache"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
set -a
. "$HOME/.s3-cod.env"
set +a
exec "/home/data/$USER/envs/low-bit-vllm/bin/python" "$repo/h200/import_vv_offload.py" \
    --target-root "/home/data/$USER/projects/low-bit-reasoning/exps/1027-fpmi-full-b" \
    --reuse-root "/home/data/$USER/projects/low-bit-reasoning/exps/1027-fpmi-w2-traces" --work-root "$root"
