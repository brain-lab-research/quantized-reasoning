#!/usr/bin/env bash
set -euo pipefail
repo=$(cd "$(dirname "$0")/.." && pwd)
root="/home/data/$USER/projects/low-bit-reasoning/exps/1029-w4-32b-full"
old="/home/data/$USER/projects/low-bit-reasoning/exps/1029-w4-full-matrix"
python="/home/data/$USER/envs/low-bit-vllm/bin/python"
export PYTHONPATH="$repo" HF_HOME="/home/data/$USER/hf-cache"
mkdir -p "$root"
if [[ -f "$root/jobs-a.txt" || -f "$root/jobs-b.txt" ]]; then
    echo ALREADY_SUBMITTED; cat "$root"/jobs-*.txt; exit 0
fi
"$python" -c 'import torch,vllm; print("STACK_READY",torch.__version__,vllm.__version__)'
"$python" "$repo/h200/w4_matrix.py" --root "$root" --prepare --sizes 32
revision="/home/data/$USER/projects/low-bit-reasoning/exps/1029-gpqa-w4-full/model-revision.json"
[[ -f "$revision" ]] || { echo MISSING_GPQA_MODEL_REVISION >&2; exit 1; }
cp -n "$revision" "$root/model-32b-revision.json"
common=(-A fpmi-p10 -p gpu --gres=gpu:1 --cpus-per-task=8 --mem=96G --time=08:00:00)
for worker in a b; do
    previous=''
    for idx in 0 1 2; do
        dep=(); [[ -z "$previous" ]] || dep=(--dependency="afterany:$previous")
        job=$(sbatch --parsable "${common[@]}" "${dep[@]}" --job-name="w4-32-$worker-$idx" --output="$root/worker-$worker-$idx.log" "$repo/h200/w4_matrix_worker.sh" "$repo" "$root" "$worker")
        printf '%s\n' "$job" >> "$root/jobs-$worker.txt"
        previous=$job
    done
    echo "WORKER_$worker $(tr '\n' ' ' < "$root/jobs-$worker.txt")"
done
# Only the specific superseded W4 queues are stopped; their saved results remain.
for worker in a b; do
    file="$old/jobs-$worker.txt"
    if [[ -f "$file" ]]; then
        while read -r job; do [[ "$job" =~ ^[0-9]+$ ]] && scancel "$job" 2>/dev/null || true; done < "$file"
    fi
done
git -C "$repo" log -1 --oneline
squeue -u "$USER" -o '%.8i %.18j %.9T %.10M %.25R'
