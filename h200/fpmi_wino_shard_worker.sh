#!/usr/bin/env bash
set -euo pipefail
repo=$1
root=$2
start=$3
count=$4
python="/home/data/$USER/envs/low-bit-vllm/bin/python"
export PYTHONPATH="$repo" PATH="$(dirname "$python"):$PATH" HF_HOME="/home/data/$USER/hf-cache"
export WANDB_MODE=offline PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 VLLM_WORKER_MULTIPROC_METHOD=spawn
unset ROCR_VISIBLE_DEVICES
cd "$repo"
started=$SECONDS
set -a; . "$HOME/.s3-cod.env"; set +a
sync_once() { "$python" "$repo/h200/sync.py" --root "$root" --prefix "shveikinda/h200/low-bit/$(basename "$root")"; }
(while sleep 180; do sync_once || true; done) & sync_pid=$!
finish() {
    rc=$?; trap - EXIT
    kill "$sync_pid" 2>/dev/null || true
    sync_once || rc=1
    if (( rc != 0 && SECONDS - started < 2400 )) && [[ -f "$root/resume-job.txt" ]]; then
        next=$(cat "$root/resume-job.txt")
        [[ "$next" == "${SLURM_JOB_ID:-}" ]] || scancel "$next" 2>/dev/null || true
    fi
    exit "$rc"
}
trap finish EXIT
trap 'exit 143' TERM INT
"$python" scripts/collect_baseline_traces.py "$root/cfg.yaml" --output "$root/artifacts" \
    --max-examples "$count" --start-index "$start" --max-tokens 32768 --max-num-seqs 32 \
    --kv-cache-gib 100 --memory-utilization 0.9 --open-thinking \
    --reuse-root "/home/data/$USER/projects/low-bit-reasoning/exps" \
    --reuse-root "/home/data/$USER/projects/low-bit-reasoning/exps/1027-fpmi-w2-traces" \
    --reuse-root "/home/data/$USER/projects/low-bit-reasoning/exps/1027-fpmi-full-b"
