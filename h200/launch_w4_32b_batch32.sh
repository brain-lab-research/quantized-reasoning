#!/usr/bin/env bash
# Prepared transition: run only after StrategyQA is complete and when requested.
set -euo pipefail
repo=$(cd "$(dirname "$0")/.." && pwd)
root="/home/data/$USER/projects/low-bit-reasoning/exps/1029-w4-32b-full"
python="/home/data/$USER/envs/low-bit-vllm/bin/python"
export PYTHONPATH="$repo" HF_HOME="/home/data/$USER/hf-cache"
for worker in a b; do
    [[ ! -e "$root/jobs-$worker-batch32.txt" ]] || { echo "ALREADY_PREPARED $worker"; exit 1; }
done
"$python" - "$root" <<'PY'
import json,sys
from pathlib import Path
from h200.paper_matrix import task_complete
root=Path(sys.argv[1])
tasks=json.loads((root/'matrix.json').read_text())
assert len(tasks)==8 and all(t['size']==32 and t['precision']=='w4' for t in tasks)
strategy=[t for t in tasks if t['benchmark']=='strategyqa']
if len(strategy)!=1 or not task_complete(strategy[0]):
    raise SystemExit('STRATEGYQA_NOT_COMPLETE: current queue was not changed')
print('STRATEGYQA_COMPLETE')
PY
common=(-A fpmi-p10 -p gpu --gres=gpu:1 --cpus-per-task=8 --mem=96G --time=08:00:00)
created=()
rollback() { rc=$?; trap - EXIT; if ((rc != 0)); then for job in "${created[@]}"; do scancel "$job" 2>/dev/null || true; done; fi; exit "$rc"; }
trap rollback EXIT
for worker in a b; do
    previous=''; ids="$root/jobs-$worker-batch32.txt"
    for idx in 0 1 2; do
        dep=(); hold=()
        if [[ -z "$previous" ]]; then hold=(--hold); else dep=(--dependency="afterany:$previous"); fi
        job=$(sbatch --parsable "${common[@]}" "${hold[@]}" "${dep[@]}" \
            --export="ALL,LOWBIT_W4_SEQS_32=32,W4_JOB_IDS_FILE=$ids" \
            --job-name="w4-b32-$worker-$idx" --output="$root/worker-$worker-b32-$idx.log" \
            "$repo/h200/w4_matrix_worker.sh" "$repo" "$root" "$worker")
        created+=("$job"); printf '%s\n' "$job" >> "$ids"; previous=$job
    done
done
# The old queues only: completed traces stay in the same experiment directory.
for worker in a b; do
    ids="$root/jobs-$worker.txt"
    if [[ -f "$ids" ]]; then
        while read -r job; do [[ "$job" =~ ^[0-9]+$ ]] && scancel "$job" 2>/dev/null || true; done < "$ids"
    fi
done
for worker in a b; do
    first=$(head -n 1 "$root/jobs-$worker-batch32.txt")
    scontrol release "$first"
    echo "BATCH32_$worker $(tr '\n' ' ' < "$root/jobs-$worker-batch32.txt")"
done
trap - EXIT
squeue -u "$USER" -o '%.8i %.18j %.9T %.10M %.25R'
