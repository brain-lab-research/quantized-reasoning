#!/usr/bin/env bash
# Queue a resumable full GPQA-Diamond W4 baseline and batch-1 wall-clock profile.
set -euo pipefail
repo=$(cd "$(dirname "$0")/.." && pwd)
root="/home/data/$USER/projects/low-bit-reasoning/exps/1029-gpqa-w4-full"
python="/home/data/$USER/envs/low-bit-vllm/bin/python"
export PYTHONPATH="$repo" HF_HOME="/home/data/$USER/hf-cache"
[[ ! -f "$root/job-ids.txt" ]] || { echo 'ALREADY_QUEUED'; cat "$root/job-ids.txt"; exit 0; }
mkdir -p "$root"
"$python" - "$repo" "$root" <<'PY'
from pathlib import Path
import sys, yaml
from quantlab.config.schema import ExperimentConfig
repo, root = map(Path, sys.argv[1:])
source = repo / 'configs/gpqa_diamond/qwen32b_gptq2bit/single_gptq2bit.yaml'
cfg = yaml.safe_load(source.read_text())
cfg['experiment_name'] = root.name
cfg['description'] = 'Full GPQA-Diamond W4 baseline; warm batch-1 latency for every question.'
cfg['benchmark'].update(name='gpqa_diamond', split='test', subset=None, max_examples=198, seed=42, start_index=0)
actor = cfg['actors'][0]
actor.update(actor_id='qwen3_32b_w4', model_id='kaitchup/Qwen3-32B-autoround-4bit-gptq', quantization_config={'bits': 4})
actor['backend_kwargs'].pop('cuda_visible_devices', None)
actor['generation_params'].update(max_new_tokens=32768, temperature=0.6, top_p=0.95, top_k=20, seed=42)
cfg['pipeline'][0].update(actor_id=actor['actor_id'], max_new_tokens=32768)
cfg['pipeline_max_total_tokens'] = 32768
cfg['output']['base_dir'] = str(root / 'artifacts')
cfg['wandb'] = {'enabled': False}
ExperimentConfig.model_validate(cfg)
path = root / 'cfg.yaml'
if path.exists() and yaml.safe_load(path.read_text()) != cfg:
    raise RuntimeError(f'Existing config differs: {path}')
path.write_text(yaml.safe_dump(cfg, sort_keys=False))
print('PREPARED', path, flush=True)
PY
common=(-A fpmi-p10 -p gpu --gres=gpu:1 --cpus-per-task=8 --mem=96G --time=08:00:00)
args=("$repo/h200/fpmi_gpqa_w4_worker.sh" "$repo" "$root")
job=$(sbatch --parsable "${common[@]}" --job-name=gpqa-w4-full --output="$root/run.log" "${args[@]}")
printf '%s\n' "$job" > "$root/job-ids.txt"
printf '%q ' sbatch "${common[@]}" --output="$root/run.log" "${args[@]}" > "$root/cmd.txt"
printf '\n' >> "$root/cmd.txt"
previous=$job
for idx in 1 2; do
    next=$(sbatch --parsable "${common[@]}" --dependency="afterany:$previous" --job-name="gpqa-w4-resume-$idx" --output="$root/resume-$idx.log" "${args[@]}")
    printf '%s\n' "$next" >> "$root/job-ids.txt"
    previous=$next
done
printf 'JOBS '; tr '\n' ' ' < "$root/job-ids.txt"; printf '\n'
git -C "$repo" log -1 --oneline
squeue -u "$USER" -o '%.8i %.20j %.9T %.10M %.25R'
