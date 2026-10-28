#!/usr/bin/env bash
set -euo pipefail
repo=$1
root=$2
w4_root=$3
python="/home/data/$USER/envs/low-bit-vllm/bin/python"
export PYTHONPATH="$repo" WANDB_MODE=offline PYTHONUNBUFFERED=1
cd "$repo"
set -a; . "$HOME/.s3-cod.env"; set +a
finish() {
    rc=$?; trap - EXIT
    "$python" "$repo/h200/sync.py" --root "$root" --prefix "shveikinda/h200/low-bit/$(basename "$root")" || rc=1
    exit "$rc"
}
trap finish EXIT
"$python" scripts/compare_gpqa_rescue_w4.py --w4 "$w4_root/artifacts" \
    --rescue "$root/artifacts" --output "$root/comparison"
echo GPQA_COMPARE_COMPLETE
