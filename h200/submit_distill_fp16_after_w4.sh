#!/usr/bin/env bash
# Queue both FP16 workers after all currently active W4 windows and a completeness gate.
set -euo pipefail
repo=$(cd "$(dirname "$0")/.." && pwd)
root="/home/data/$USER/projects/low-bit-reasoning/exps/1029-distill-llama-fp16-full"
w4="/home/data/$USER/projects/low-bit-reasoning/exps/1029-w4-32b-full"
python="/home/data/$USER/envs/low-bit-vllm/bin/python"
export PYTHONPATH="$repo" HF_HOME="/home/data/$USER/hf-cache"
mkdir -p "$root"
if [[ -f "$root/gate-job.txt" || -f "$root/jobs-a.txt" || -f "$root/jobs-b.txt" ]]; then
    echo ALREADY_SUBMITTED; cat "$root"/gate-job.txt "$root"/jobs-*.txt 2>/dev/null || true; exit 0
fi
"$python" -c 'import torch,vllm; print("STACK_READY",torch.__version__,vllm.__version__)'
"$python" "$repo/vv/distill_llama_matrix.py" --root "$root" --prepare --precision fp16
printf '%s\n' '{"model_id":"deepseek-ai/DeepSeek-R1-Distill-Llama-8B","revision":"6a6f4aa4197940add57724a7707d069478df56b1"}' > "$root/model-revision.json"
active=()
for worker in a b; do
    ids="$w4/jobs-$worker-batch32.txt"
    [[ -f "$ids" ]] || { echo "MISSING_W4_QUEUE $ids" >&2; exit 1; }
    while read -r job; do
        [[ "$job" =~ ^[0-9]+$ ]] || { echo INVALID_W4_ID >&2; exit 1; }
        if [[ -n "$(squeue -h -j "$job" -o '%i')" ]]; then active+=("$job"); fi
    done < "$ids"
done
dep=()
if ((${#active[@]})); then joined=$(IFS=:; echo "${active[*]}"); dep=(--dependency="afterany:$joined"); fi
created=()
rollback() { rc=$?; trap - EXIT; if ((rc != 0)); then for job in "${created[@]}"; do scancel "$job" 2>/dev/null || true; done; fi; exit "$rc"; }
trap rollback EXIT
gate=$(sbatch --parsable --hold -A fpmi-p10 -p cpu --cpus-per-task=1 --mem=2G --time=00:10:00 \
    "${dep[@]}" --job-name=distill-w4-gate --output="$root/gate.log" "$repo/h200/distill_fp16_gate.sh" "$repo")
created+=("$gate"); printf '%s\n' "$gate" > "$root/gate-job.txt"
common=(-A fpmi-p10 -p gpu --gres=gpu:1 --cpus-per-task=8 --mem=96G --time=08:00:00)
for worker in a b; do
    previous=''
    for idx in 0 1 2; do
        if [[ -z "$previous" ]]; then dependency="afterok:$gate"; else dependency="afterany:$previous"; fi
        job=$(sbatch --parsable "${common[@]}" --dependency="$dependency" --job-name="distill-$worker-$idx" \
            --output="$root/worker-$worker-$idx.log" "$repo/h200/distill_fp16_worker.sh" "$repo" "$root" "$worker")
        created+=("$job"); printf '%s\n' "$job" >> "$root/jobs-$worker.txt"; previous=$job
    done
    echo "WORKER_$worker $(tr '\n' ' ' < "$root/jobs-$worker.txt")"
done
scontrol release "$gate"
trap - EXIT
echo "GATE=$gate W4_ACTIVE=${active[*]:-none}"
git -C "$repo" log -1 --oneline
squeue -u "$USER" -o '%.8i %.20j %.9T %.10M %.25R'
