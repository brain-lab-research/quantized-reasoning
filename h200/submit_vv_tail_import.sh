#!/usr/bin/env bash
set -euo pipefail
repo=$(cd "$(dirname "$0")/.." && pwd)
root="/home/data/$USER/projects/low-bit-reasoning/exps/1028-fpmi-tail-import"
if [[ -f "$root/job-id.txt" ]]; then
    printf 'ALREADY_QUEUED=%s\n' "$(cat "$root/job-id.txt")"
    exit 0
fi
mkdir -p "$root"
command=(sbatch --parsable -A fpmi-p10 -p cpu --cpus-per-task=1 --mem=8G --time=08:00:00
    --job-name=lowbit-tail-import --output="$root/run.log" "$repo/h200/vv_import_worker.sh" "$repo" "$root" tail)
job=$("${command[@]}")
printf '%s\n' "$job" > "$root/job-id.txt"
printf '%q ' "${command[@]}" > "$root/cmd.txt"
printf '\n' >> "$root/cmd.txt"
printf 'TAIL_IMPORT_JOB=%s\n' "$job"
squeue -j "$job" -o '%.8i %.20j %.9T %.10M %.25R'
