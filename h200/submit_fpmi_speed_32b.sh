#!/usr/bin/env bash
# Prepare three paired profiles and queue them on an available GPU.
set -euo pipefail
repo=$(cd "$(dirname "$0")/.." && pwd)
root="/home/data/$USER/projects/low-bit-reasoning/exps/1028-speed-32b"
python="/home/data/$USER/envs/low-bit-vllm/bin/python"
export PYTHONPATH="$repo" HF_HOME="/home/data/$USER/hf-cache"
previous=${1:-0}
[[ $previous =~ ^[0-9]+$ ]] || exit 2
dependency=()
if [[ "$previous" != 0 ]]; then dependency=(--dependency="afterany:$previous"); fi
[[ ! -f "$root/job-id.txt" ]] || { echo "ALREADY_QUEUED $(cat "$root/job-id.txt")"; exit 0; }
mkdir -p "$root"/{w2,w4,fp16}
"$python" - "$repo" "$root" <<'PY'
import copy, sys
from pathlib import Path
import yaml
from quantlab.config.schema import ExperimentConfig
repo, root = map(Path, sys.argv[1:])
for mode in ('w2', 'w4', 'fp16'):
    group = 'qwen32b_fp16' if mode == 'fp16' else 'qwen32b_gptq2bit'
    name = 'single_fp16.yaml' if mode == 'fp16' else 'single_gptq2bit.yaml'
    cfg = yaml.safe_load((repo / 'configs' / 'math500' / group / name).read_text())
    cfg['experiment_name'] = f'1028-speed-32b-{mode}'
    cfg['benchmark'].update(name='math500', split='test', max_examples=20, seed=42, start_index=0)
    actor = cfg['actors'][0]
    if mode == 'w4':
        actor['model_id'] = 'kaitchup/Qwen3-32B-autoround-4bit-gptq'
        actor['quantization_config'] = {'bits': 4}
    actor['backend_kwargs'].pop('cuda_visible_devices', None)
    actor['generation_params'].update(max_new_tokens=32768, temperature=0.6, top_p=0.95, top_k=20, seed=42)
    cfg['pipeline'][0]['max_new_tokens'] = 32768
    cfg['pipeline_max_total_tokens'] = 32768
    cfg['output']['base_dir'] = str(root / mode / 'artifacts')
    cfg['wandb'] = {'enabled': False}
    ExperimentConfig.model_validate(cfg)
    path = root / mode / 'cfg.yaml'
    if path.exists() and yaml.safe_load(path.read_text()) != cfg:
        raise RuntimeError(f'Existing config differs: {path}')
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    print('PREPARED', path)
PY
common=(-A fpmi-p10 -p gpu --gres=gpu:1 --cpus-per-task=8 --mem=176G --time=08:00:00)
args=("$repo/h200/fpmi_speed_32b_worker.sh" "$repo" "$root")
job=$(sbatch --parsable "${common[@]}" "${dependency[@]}" --job-name=lowbit-speed-32b --output="$root/run.log" "${args[@]}")
printf '%s\n' "$job" > "$root/job-id.txt"
printf '%q ' sbatch "${common[@]}" "${dependency[@]}" --output="$root/run.log" "${args[@]}" > "$root/cmd.txt"
printf '\n' >> "$root/cmd.txt"
resume=$(sbatch --parsable "${common[@]}" --dependency="afterany:$job" --job-name=lowbit-speed-resume --output="$root/resume.log" "${args[@]}")
printf '%s\n' "$resume" > "$root/resume-job.txt"
printf 'SPEED_JOB=%s RESUME=%s AFTER=%s\n' "$job" "$resume" "$previous"
git -C "$repo" log -1 --oneline
squeue -u "$USER" -o '%.8i %.20j %.9T %.10M %.25R'
