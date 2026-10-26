#!/usr/bin/env bash
# Run under tmux; caller sets CUDA_VISIBLE_DEVICES after checking the shared GPU.
set -euo pipefail
repo=$1
root=$2
backup=${3:-local-only}
. /data/users/shveikinda/miniconda3/etc/profile.d/conda.sh
conda activate base
cd "$repo"
export WANDB_MODE=offline PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
unset ROCR_VISIBLE_DEVICES
ql_python=$(command -v python)
mkdir -p "$root"
git rev-parse HEAD > "$root/code-revision.txt"
printf 'START host=%s devices=%s backup=%s\n' "$(hostname)" "$CUDA_VISIBLE_DEVICES" "$backup"
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv
"$ql_python" -c 'import torch,vllm,transformers; print("STACK",torch.__version__,vllm.__version__,transformers.__version__)'
sync_once() {
    if [[ "$backup" == s3 ]]; then
        /usr/bin/python3 "$repo/h200/sync.py" --root "$root" --prefix "shveikinda/vv/low-bit/$(basename "$root")"
    fi
}
if [[ "$backup" == s3 ]]; then
    set -a
    . "$HOME/.s3-cod.env"
    set +a
    sync_once
fi
(while sleep 180; do sync_once || printf 'BACKUP_RETRY\n'; done) &
sync_pid=$!
finish() {
    local rc=$?
    trap - EXIT
    kill "$sync_pid" 2>/dev/null || true
    sync_once || { printf 'FINAL_BACKUP_FAILED\n'; rc=1; }
    printf 'JOB_EXIT %s\n' "$rc"
    exit "$rc"
}
trap finish EXIT
trap 'exit 143' TERM INT
"$ql_python" "$repo/h200/prepare.py" --root "$root" --memory-utilization 0.25 --batch-size 4
for precision in fp16 w2; do
    "$ql_python" "$repo/h200/smoke.py" --precision "$precision" --output "$root/smoke-$precision.json" --memory-utilization 0.25 --batch-size 4 > "$root/smoke-$precision.log" 2>&1
    printf 'SMOKE_OK %s\n' "$precision"
    sync_once
done
"$ql_python" "$repo/h200/collect.py" --root "$root" --hours 0
