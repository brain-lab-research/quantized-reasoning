#!/usr/bin/env bash
# Download 32B W2 within the agreed disk cap, then queue smoke and pilots.
set -euo pipefail
repo=$(cd "$(dirname "$0")/.." && pwd)
root="/home/data/$USER/projects/low-bit-reasoning/exps/1028-reproduction-pilot"
python="/home/data/$USER/envs/low-bit-vllm/bin/python"
export PYTHONPATH="$repo" HF_HOME="/home/data/$USER/hf-cache" WANDB_MODE=offline
mkdir -p "$root"
if [[ -e "$root/smoke-job.txt" || -e "$root/job-id.txt" ]]; then
    echo 'ALREADY_QUEUED: inspect recorded job IDs before retrying' >&2
    exit 1
fi
used=$(du -s --block-size=1 "/home/data/$USER" | cut -f1)
# Exact public snapshot is 13.14 GB; reserve 2 GiB extra for cache metadata and logs.
if (( used + 13140044418 + 2147483648 > 390 * 1024 * 1024 * 1024 )); then
    echo "DISK_CAP_BLOCK used_bytes=$used" >&2
    exit 1
fi
"$python" - <<'PY'
from huggingface_hub import snapshot_download
path = snapshot_download('kaitchup/Qwen3-32B-autoround-2bit-gptq',
                         allow_patterns=['*.json', '*.safetensors', '*.txt', '*.jinja'])
print('MODEL_READY', path, flush=True)
PY
bash "$repo/h200/reproduction_pilot.sh" prepare
common=(-A fpmi-p10 -p gpu --gres=gpu:1 --cpus-per-task=8 --mem=64G --time=08:00:00)
previous=${1:?Pass the current 8B worker job ID}
[[ $previous =~ ^[0-9]+$ ]] || { echo 'Invalid previous job ID' >&2; exit 2; }
smoke=$(sbatch --parsable "${common[@]}" --dependency="afterany:$previous" --job-name=lowbit-pilot-smoke --output="$root/smoke.log" "$repo/h200/reproduction_pilot.sh" smoke)
printf '%s\n' "$smoke" > "$root/smoke-job.txt"
job=$(sbatch --parsable "${common[@]}" --dependency="afterok:$smoke" --job-name=lowbit-pilot --output="$root/run.log" "$repo/h200/reproduction_pilot.sh" worker)
printf '%s\n' "$job" > "$root/job-id.txt"
next=$(sbatch --parsable "${common[@]}" --dependency="afterany:$job" --job-name=lowbit-pilot-resume --output="$root/resume.log" "$repo/h200/reproduction_pilot.sh" worker)
printf '%s\n' "$next" > "$root/resume-job.txt"
printf 'sbatch -A fpmi-p10 -p gpu --gres=gpu:1 --dependency=afterok:%s %s worker\n' "$smoke" "$repo/h200/reproduction_pilot.sh" > "$root/cmd.txt"
printf 'SMOKE=%s PILOT=%s RESUME=%s AFTER=%s\n' "$smoke" "$job" "$next" "$previous"
squeue -u "$USER" -o '%.8i %.20j %.9T %.10M %.25R'
