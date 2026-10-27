#!/usr/bin/env bash
set -euo pipefail
repo=$1
root=$2
mode=$3
shift 3
base="/home/data/$USER/projects/low-bit-reasoning/runtime"
ql_python="/home/data/$USER/envs/low-bit-vllm/bin/python"
export PYTHONPATH="$repo" PATH="$(dirname "$ql_python"):$PATH"
export HF_HOME="/home/data/$USER/hf-cache"
export TMPDIR="/home/data/$USER/tmp"
export WANDB_MODE=offline PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 VLLM_WORKER_MULTIPROC_METHOD=spawn
unset ROCR_VISIBLE_DEVICES
case "${CUDA_HOME:-}" in *cuda-shim*) unset CUDA_HOME ;; esac
started=$SECONDS
mkdir -p "$TMPDIR"
cd "$repo"
printf 'START job=%s mode=%s devices=%s\n' "$SLURM_JOB_ID" "$mode" "$CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=uuid,memory.total --format=csv
set -a
. "$HOME/.s3-cod.env"
set +a
sync_once() { "$ql_python" "$repo/h200/sync.py" --root "$root" --prefix "shveikinda/h200/low-bit/$(basename "$root")"; }
(while sleep 180; do sync_once || printf 'BACKUP_RETRY\n'; done) &
sync_pid=$!
finish() {
    rc=$?
    trap - EXIT
    kill "$sync_pid" 2>/dev/null || true
    sync_once || rc=1
    if [[ "$mode" == smoke && "$rc" != 0 ]]; then
        for file in job-id.txt resume-job.txt; do
            [[ -f "$root/$file" ]] && scancel "$(cat "$root/$file")" || true
        done
    elif [[ ( "$mode" == full || "$mode" == all ) && -f "$root/resume-job.txt" ]]; then
        next=$(cat "$root/resume-job.txt")
        if [[ "$next" != "$SLURM_JOB_ID" ]] && { [[ "$rc" == 0 ]] || (( SECONDS-started < 2400 )); }; then
            scancel "$next" || true
        fi
    fi
    exit "$rc"
}
trap finish EXIT
trap 'exit 143' TERM INT
if [[ "$mode" == smoke ]]; then
    "$ql_python" "$repo/h200/smoke.py" --precision w2 --output "$root/smoke-w2.json" --memory-utilization 0.75 --batch-size 16 --kv-cache-gib 80
elif [[ "$mode" == all ]]; then
    "$ql_python" "$repo/h200/paper_matrix.py" --root "$root" --sizes 8 --precisions w2 --full-benchmarks --seqs-8 32 --kv-cache-gib 100 --memory-utilization 0.9 "$@"
elif [[ "$mode" == profile ]]; then
    "$ql_python" "$repo/scripts/profile_baseline_latency.py" --config "$root/cfg.yaml" --output "$root/artifacts" --examples 20
else
    "$ql_python" "$repo/h200/paper_matrix.py" --root "$root" --sizes 8 --precisions w2 --full-first --kv-cache-gib 80 --memory-utilization 0.75
fi
