#!/usr/bin/env bash
# Invoked by submit.sh; all outputs/configs live outside the infrastructure repo.
set -euo pipefail
repo=$1
root=$2
ql_python=$3
export WANDB_MODE=offline PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export HF_HOME="/home/data/$USER/hf-cache"
export TMPDIR="/home/data/$USER/tmp"
mkdir -p "$TMPDIR"
unset ROCR_VISIBLE_DEVICES
export PATH="$(dirname "$ql_python"):$PATH"
cd "$repo"
git rev-parse HEAD > "$root/code-revision.txt"
printf 'START job=%s host=%s devices=%s\n' "$SLURM_JOB_ID" "$(hostname)" "$CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=name,memory.total --format=csv
"$ql_python" -c 'import torch,vllm,transformers; print("STACK",torch.__version__,vllm.__version__,transformers.__version__)'
set -a
. "$HOME/.s3-cod.env"
set +a
sync_python=${QL_SYNC_PYTHON:-$ql_python}
"$sync_python" -c 'import boto3'
prefix="shveikinda/h200/low-bit/$QL_SERIES"
sync_once() { "$sync_python" "$repo/h200/sync.py" --root "$root" --prefix "$prefix"; }
sync_once
(while sleep 180; do sync_once || printf 'BACKUP_RETRY\n'; done) &
sync_pid=$!
finish() {
    local rc=$?
    trap - EXIT
    kill "$sync_pid" 2>/dev/null || true
    sync_once || { printf 'FINAL_BACKUP_FAILED\n'; rc=1; }
    exit "$rc"
}
trap finish EXIT
trap 'exit 143' TERM INT
"$ql_python" "$repo/h200/prepare.py" --root "$root"
for precision in fp16 w2; do
    "$ql_python" "$repo/h200/smoke.py" --precision "$precision" --output "$root/smoke-$precision.json" > "$root/smoke-$precision.log" 2>&1
    printf 'SMOKE_OK %s\n' "$precision"
    sync_once
done
"$ql_python" "$repo/h200/collect.py" --root "$root" --hours 7.5
