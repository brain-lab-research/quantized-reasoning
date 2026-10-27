#!/usr/bin/env bash
# Useful GPU work after each complete baseline chain, not a repeat of saved traces.
set -euo pipefail
repo=$(cd "$(dirname "$0")/.." && pwd)
ql_python="/home/data/$USER/envs/low-bit-vllm/bin/python"
for label in a b; do
    baseline="/home/data/$USER/projects/low-bit-reasoning/exps/1027-fpmi-full-$label"
    benchmark=math500
    [[ "$label" == b ]] && benchmark=gsm8k
    root="/home/data/$USER/projects/low-bit-reasoning/exps/1028-fpmi-latency-$benchmark"
    if [[ -f "$root/job-id.txt" ]]; then
        printf 'ALREADY_QUEUED_%s=%s\n' "$label" "$(cat "$root/job-id.txt")"
        continue
    fi
    previous=$(cat "$baseline/extra-job.txt")
    mkdir -p "$root"
    source="$baseline/1027-paper-qwen8b-${benchmark}-w2-all/cfg.yaml"
    "$ql_python" - "$source" "$root" <<'PY'
import sys
from pathlib import Path
import yaml
source, root = Path(sys.argv[1]), Path(sys.argv[2])
raw = yaml.safe_load(source.read_text())
raw['experiment_name'] = root.name
raw['description'] = 'Warm batch-1 wall-clock profile: 20 fixed cases, full token budget.'
raw['benchmark']['max_examples'] = 20
raw['output']['base_dir'] = str(root/'artifacts')
(root/'cfg.yaml').write_text(yaml.safe_dump(raw, sort_keys=False))
PY
    command=(sbatch --parsable -A fpmi-p10 -p gpu --gres=gpu:1 --cpus-per-task=4 --mem=32G
        --time=08:00:00 --dependency="afterany:$previous" --job-name="lowbit-latency-$label"
        --output="$root/run.log" "$repo/h200/fpmi_worker.sh" "$repo" "$root" profile)
    job=$("${command[@]}")
    printf '%s\n' "$job" > "$root/job-id.txt"
    printf '%q ' "${command[@]}" > "$root/cmd.txt"
    printf '\n' >> "$root/cmd.txt"
    printf 'PROFILE_%s=%s AFTER=%s\n' "$label" "$job" "$previous"
done
git -C "$repo" log -1 --oneline
squeue -u "$USER" -o '%.8i %.20j %.9T %.10M %.25R'
