#!/usr/bin/env bash
# Add one more resumable window without stopping or modifying running jobs.
set -euo pipefail
repo=$(cd "$(dirname "$0")/.." && pwd)
old="/home/data/$USER/projects/low-bit-reasoning/exps/1027-fpmi-w2-traces"
for label in a b; do
    root="/home/data/$USER/projects/low-bit-reasoning/exps/1027-fpmi-full-$label"
    if [[ -f "$root/extra-job.txt" ]]; then
        printf 'ALREADY_QUEUED_%s=%s\n' "$label" "$(cat "$root/extra-job.txt")"
        continue
    fi
    previous=$(cat "$root/resume-job.txt")
    if [[ "$label" == a ]]; then
        benchmarks=(math500 gsm8k aime2026)
    else
        benchmarks=(gpqa_diamond arc_easy arc_challenge piqa strategyqa winogrande)
    fi
    command=(sbatch --parsable -A fpmi-p10 -p gpu --gres=gpu:1 --cpus-per-task=8 --mem=64G
        --time=08:00:00 --dependency="afterany:$previous" --job-name="lowbit-extra-$label"
        --output="$root/extra.log" "$repo/h200/fpmi_worker.sh" "$repo" "$root" all
        --reuse-root "$old" --benchmarks "${benchmarks[@]}")
    job=$("${command[@]}")
    printf '%s\n' "$job" > "$root/extra-job.txt"
    printf '%q ' "${command[@]}" > "$root/extra-cmd.txt"
    printf '\n' >> "$root/extra-cmd.txt"
    printf 'EXTRA_%s=%s AFTER=%s\n' "$label" "$job" "$previous"
done
git -C "$repo" log -1 --oneline
squeue -u "$USER" -o '%.8i %.18j %.9T %.10M %.25R'
