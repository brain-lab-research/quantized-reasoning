#!/usr/bin/env bash
# Replace two overlapping 8B W2 workers with disjoint WinoGrande shards.
set -euo pipefail
repo=$(cd "$(dirname "$0")/.." && pwd)
python="/home/data/$USER/envs/low-bit-vllm/bin/python"
project="/home/data/$USER/projects/low-bit-reasoning/exps"
source="$project/1027-fpmi-w2-traces/1027-paper-qwen8b-winogrande-w2-all/cfg.yaml"
pilot="$project/1028-reproduction-pilot"
export PYTHONPATH="$repo"
old_a=${1:?Pass the running spare Wino job ID}
old_b=${2:?Pass the running second worker job ID}
[[ $old_a =~ ^[0-9]+$ && $old_b =~ ^[0-9]+$ && $old_a != "$old_b" ]] || exit 2
for job in "$old_a" "$old_b"; do
    [[ -n $(squeue -h -j "$job" -t RUNNING -o %i) ]] || { echo "NOT_RUNNING $job" >&2; exit 1; }
done
[[ -f "$pilot/smoke-job.txt" ]] || { echo 'PILOT_SETUP_NOT_FINISHED' >&2; exit 1; }
smoke=$(cat "$pilot/smoke-job.txt")
for job in "$smoke" 17580 17088; do
    [[ -n $(squeue -h -j "$job" -t PENDING -o %i) ]] || { echo "NOT_PENDING $job" >&2; exit 1; }
done
roots=("$project/1028-wino-shard-a" "$project/1028-wino-shard-b")
for root in "${roots[@]}"; do
    [[ ! -e "$root/job-id.txt" ]] || { echo "ALREADY_QUEUED $root" >&2; exit 1; }
    mkdir -p "$root"
done
"$python" - "$source" "${roots[@]}" <<'PY'
import copy, sys
from pathlib import Path
import yaml
from quantlab.config.schema import ExperimentConfig
source, *roots = map(Path, sys.argv[1:])
base = yaml.safe_load(source.read_text())
for root, start, count in zip(roots, (0, 800), (800, 467)):
    cfg = copy.deepcopy(base)
    cfg['experiment_name'] = root.name
    cfg['benchmark'].update(start_index=start, max_examples=count)
    cfg['output']['base_dir'] = str(root / 'artifacts')
    ExperimentConfig.model_validate(cfg)
    path = root / 'cfg.yaml'
    if path.exists() and yaml.safe_load(path.read_text()) != cfg:
        raise RuntimeError(f'Existing config differs: {path}')
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    print('PREPARED', path, start, count)
PY
# Hold only the existing follow-ups while their dependencies are replaced.
held=("$smoke" 17580 17088)
submitted=()
cleanup() {
    rc=$?; trap - EXIT
    if (( rc != 0 )); then
        for job in "${submitted[@]}"; do scancel "$job" 2>/dev/null || true; done
        for job in "${held[@]}"; do scontrol release "$job" 2>/dev/null || true; done
    fi
    exit "$rc"
}
trap cleanup EXIT
for job in "${held[@]}"; do scontrol hold "$job"; done
common=(-A fpmi-p10 -p gpu --gres=gpu:1 --cpus-per-task=8 --mem=64G --time=08:00:00)
resumes=()
for idx in 0 1; do
    root=${roots[$idx]}
    if (( idx == 0 )); then start=0; count=800; previous=$old_a; else start=800; count=467; previous=$old_b; fi
    args=("$repo/h200/fpmi_wino_shard_worker.sh" "$repo" "$root" "$start" "$count")
    job=$(sbatch --parsable "${common[@]}" --dependency="afterany:$previous" --job-name="wino-shard-$idx" --output="$root/run.log" "${args[@]}")
    submitted+=("$job")
    printf '%s\n' "$job" > "$root/job-id.txt"
    printf '%q ' sbatch "${common[@]}" --dependency="afterany:$previous" --output="$root/run.log" "${args[@]}" > "$root/cmd.txt"
    printf '\n' >> "$root/cmd.txt"
    resume=$(sbatch --parsable "${common[@]}" --dependency="afterany:$job" --job-name="wino-shard-resume-$idx" --output="$root/resume.log" "${args[@]}")
    submitted+=("$resume"); resumes+=("$resume")
    printf '%s\n' "$resume" > "$root/resume-job.txt"
done
dependency="afterany:${resumes[0]}:${resumes[1]}"
scontrol update "JobId=$smoke" "Dependency=$dependency"
scontrol update "JobId=17088" "Dependency=$dependency"
# The old resume would otherwise restart the overlapping full benchmark.
scancel 17580
held=("$smoke" 17088)
for job in "${held[@]}"; do scontrol release "$job"; done
held=()
# Only completed records are reused; in-flight requests may be regenerated.
scancel "$old_a" "$old_b"
trap - EXIT
printf 'SHARD_A=%s RESUME_A=%s SHARD_B=%s RESUME_B=%s PILOT_SMOKE=%s\n' "${submitted[0]}" "${resumes[0]}" "${submitted[2]}" "${resumes[1]}" "$smoke"
squeue -u "$USER" -o '%.8i %.20j %.9T %.10M %.25R'
