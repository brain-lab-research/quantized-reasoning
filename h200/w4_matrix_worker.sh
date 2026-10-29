#!/usr/bin/env bash
set -euo pipefail
repo=$1; root=$2; worker=$3
python="/home/data/$USER/envs/low-bit-vllm/bin/python"
export PYTHONPATH="$repo" PATH="$(dirname "$python"):$PATH" HF_HOME="/home/data/$USER/hf-cache"
export WANDB_MODE=offline PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 VLLM_WORKER_MULTIPROC_METHOD=spawn
unset ROCR_VISIBLE_DEVICES
cd "$repo"
started=$SECONDS
stage=$(mktemp -d "/dev/shm/lowbit-w4-${SLURM_JOB_ID:?}-XXXXXXXX")
set -a; . "$HOME/.s3-cod.env"; set +a
sync_once() { "$python" "$repo/h200/sync.py" --root "$root" --prefix "shveikinda/h200/low-bit/$(basename "$root")"; }
(while sleep 180; do sync_once || echo BACKUP_RETRY; done) & sync_pid=$!
finish() {
    rc=$?; trap - EXIT
    kill "$sync_pid" 2>/dev/null || true
    sync_once || rc=1
    case "$stage" in /dev/shm/lowbit-w4-"$SLURM_JOB_ID"-*) rm -rf -- "$stage" ;; *) rc=1 ;; esac
    if (( rc != 0 && SECONDS-started < 2400 )) && [[ -f "$root/jobs-$worker.txt" ]]; then
        while read -r job; do [[ "$job" == "$SLURM_JOB_ID" ]] || scancel "$job" 2>/dev/null || true; done < "$root/jobs-$worker.txt"
    fi
    exit "$rc"
}
trap finish EXIT
trap 'exit 143' TERM INT
echo "START job=$SLURM_JOB_ID worker=$worker device=$CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=uuid,memory.total --format=csv
"$python" "$repo/h200/w4_matrix.py" --root "$root" --worker "$worker" --stage "$stage"
echo WORKER_COMPLETE
