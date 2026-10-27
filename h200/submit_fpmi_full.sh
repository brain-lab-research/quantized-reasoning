#!/usr/bin/env bash
# Replace the partial 8B W2 queue with two disjoint full-benchmark workers.
set -euo pipefail
repo=$(cd "$(dirname "$0")/.." && pwd)
old="/home/data/$USER/projects/low-bit-reasoning/exps/1027-fpmi-w2-traces"
python="/home/data/$USER/envs/low-bit-vllm/bin/python"
export PYTHONPATH="$repo" HF_HOME="/home/data/$USER/hf-cache"
"$python" -c 'import torch,vllm; print("STACK_READY", torch.__version__, vllm.__version__)'
for label in a b; do
    root="/home/data/$USER/projects/low-bit-reasoning/exps/1027-fpmi-full-$label"
    if [[ -f "$root/job-id.txt" ]]; then
        echo "ALREADY_SUBMITTED $root; inspect existing jobs before resubmitting"
        exit 1
    fi
    mkdir -p "$root"
    "$python" "$repo/h200/paper_matrix.py" --root "$root" --full-benchmarks --prepare-only --sizes 8 --precisions w2
done
# Save each finished trace; only unfinished requests are regenerated.
if [[ -f "$old/resume-job.txt" ]]; then scancel "$(cat "$old/resume-job.txt")" 2>/dev/null || true; fi
if [[ -f "$old/job-id.txt" ]]; then
    previous=$(cat "$old/job-id.txt")
    scancel "$previous" 2>/dev/null || true
    dependency=(--dependency="afterany:$previous")
else
    dependency=()
fi
for label in a b; do
    root="/home/data/$USER/projects/low-bit-reasoning/exps/1027-fpmi-full-$label"
    if [[ "$label" == a ]]; then
        benchmarks=(math500 gsm8k aime2026)
    else
        benchmarks=(gpqa_diamond arc_easy arc_challenge piqa strategyqa winogrande)
    fi
    command=("$repo/h200/fpmi_worker.sh" "$repo" "$root" all --reuse-root "$old" --benchmarks "${benchmarks[@]}")
    common=(-A fpmi-p10 -p gpu --gres=gpu:1 --cpus-per-task=8 --mem=64G --time=08:00:00)
    job=$(sbatch --parsable "${common[@]}" "${dependency[@]}" --job-name="lowbit-full-$label" --output="$root/run.log" "${command[@]}")
    printf '%s\n' "$job" > "$root/job-id.txt"
    printf '%q ' sbatch "${common[@]}" "${dependency[@]}" --output="$root/run.log" "${command[@]}" > "$root/cmd.txt"
    printf '\n' >> "$root/cmd.txt"
    next=$(sbatch --parsable "${common[@]}" --dependency="afterany:$job" --job-name="lowbit-resume-$label" --output="$root/resume.log" "${command[@]}")
    printf '%s\n' "$next" > "$root/resume-job.txt"
    printf 'FULL_%s=%s RESUME_%s=%s\n' "$label" "$job" "$label" "$next"
done
git -C "$repo" log -1 --oneline
squeue -u "$USER" -o '%.8i %.18j %.9T %.20b %.30R'
