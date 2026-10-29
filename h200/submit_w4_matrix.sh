#!/usr/bin/env bash
set -euo pipefail
repo=$(cd "$(dirname "$0")/.." && pwd)
root="/home/data/$USER/projects/low-bit-reasoning/exps/1029-w4-full-matrix"
python="/home/data/$USER/envs/low-bit-vllm/bin/python"
export PYTHONPATH="$repo" HF_HOME="/home/data/$USER/hf-cache"
mkdir -p "$root"
if [[ -f "$root/jobs-a.txt" || -f "$root/jobs-b.txt" ]]; then
    echo ALREADY_SUBMITTED; cat "$root"/jobs-*.txt; exit 0
fi
"$python" -c 'import torch,vllm; print("STACK_READY",torch.__version__,vllm.__version__)'
"$python" "$repo/h200/w4_matrix.py" --root "$root" --prepare
common=(-A fpmi-p10 -p gpu --gres=gpu:1 --cpus-per-task=8 --mem=96G --time=08:00:00)
for worker in a b; do
    previous=''
    for idx in 0 1 2; do
        dep=(); [[ -z "$previous" ]] || dep=(--dependency="afterany:$previous")
        log="$root/worker-$worker-$idx.log"
        job=$(sbatch --parsable "${common[@]}" "${dep[@]}" --job-name="w4-$worker-$idx" --output="$log" "$repo/h200/w4_matrix_worker.sh" "$repo" "$root" "$worker")
        printf '%s\n' "$job" >> "$root/jobs-$worker.txt"
        previous=$job
    done
    echo "WORKER_$worker $(tr '\n' ' ' < "$root/jobs-$worker.txt")"
done
git -C "$repo" log -1 --oneline
squeue -u "$USER" -o '%.8i %.18j %.9T %.10M %.25R'
