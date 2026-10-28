#!/usr/bin/env bash
# Use the free FPMI card for the remaining full 8B W2 WinoGrande run.
set -euo pipefail
repo=$(cd "$(dirname "$0")/.." && pwd)
root="/home/data/$USER/projects/low-bit-reasoning/exps/1027-fpmi-w2-traces"
if [[ -f "$root/spare-wino-job.txt" ]]; then
    printf 'ALREADY_QUEUED=%s\n' "$(cat "$root/spare-wino-job.txt")"
    exit 0
fi
command=(sbatch --parsable -A fpmi-p10 -p gpu --gres=gpu:1 --cpus-per-task=8 --mem=64G
    --time=08:00:00 --job-name=lowbit-wino-spare --output="$root/spare-wino.log"
    "$repo/h200/fpmi_worker.sh" "$repo" "$root" all --reuse-root "$root" --benchmarks winogrande)
job=$("${command[@]}")
printf '%s\n' "$job" > "$root/spare-wino-job.txt"
printf '%q ' "${command[@]}" > "$root/spare-wino-cmd.txt"
printf '\n' >> "$root/spare-wino-cmd.txt"
printf 'SPARE_WINO_JOB=%s\n' "$job"
squeue -j "$job" -o '%.8i %.20j %.9T %.10M %.25R'
