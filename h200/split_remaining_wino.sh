#!/usr/bin/env bash
# Replace the active WinoGrande tail with two shards balanced by unsolved IDs.
set -euo pipefail
repo=$(cd "$(dirname "$0")/.." && pwd)
python="/home/data/$USER/envs/low-bit-vllm/bin/python"
project="/home/data/$USER/projects/low-bit-reasoning/exps"
old_root="$project/1028-wino-shard-b"
pilot="$project/1028-reproduction-pilot"
export PYTHONPATH="$repo"
old_job=${1:?Pass the active WinoGrande tail job ID}
old_resume=${2:?Pass its pending resume job ID}
[[ $old_job =~ ^[0-9]+$ && $old_resume =~ ^[0-9]+$ ]] || exit 2
[[ -n $(squeue -h -j "$old_job" -t RUNNING -o %i) ]] || { echo "NOT_RUNNING $old_job" >&2; exit 1; }
smoke=$(cat "$pilot/smoke-job.txt")
for job in "$old_resume" "$smoke" 17088; do
    [[ -n $(squeue -h -j "$job" -t PENDING -o %i) ]] || { echo "NOT_PENDING $job" >&2; exit 1; }
done
roots=("$project/1028-wino-remaining-a" "$project/1028-wino-remaining-b")
for root in "${roots[@]}"; do
    [[ ! -e "$root/job-id.txt" ]] || { echo "ALREADY_QUEUED $root" >&2; exit 1; }
    mkdir -p "$root"
done
"$python" - "$old_root" "${roots[@]}" <<'PY'
import copy, json, sys
from pathlib import Path
import yaml
from quantlab.config.schema import ExperimentConfig
old, *roots = map(Path, sys.argv[1:])
cfg = yaml.safe_load((old / 'cfg.yaml').read_text())
assert cfg['benchmark']['start_index'] == 800 and cfg['benchmark']['max_examples'] == 467
art = old / 'artifacts'
run = (art / 'run-id.txt').read_text().strip()
rows = [json.loads(x) for x in (art / run / 'judgements.jsonl').read_text().splitlines() if x.strip()]
ids = [int(x['example_id'].rsplit('_', 1)[-1]) for x in rows]
assert len(ids) == len(set(ids)) and set(ids) <= set(range(800, 1267))
missing = sorted(set(range(800, 1267)) - set(ids))
if len(missing) < 4:
    raise RuntimeError(f'Too few unfinished cases to split: {len(missing)}')
mid = missing[len(missing) // 2]
for root, start, count in zip(roots, (800, mid), (mid - 800, 1267 - mid)):
    raw = copy.deepcopy(cfg)
    raw['experiment_name'] = root.name
    raw['benchmark'].update(start_index=start, max_examples=count)
    raw['output']['base_dir'] = str(root / 'artifacts')
    ExperimentConfig.model_validate(raw)
    path = root / 'cfg.yaml'
    if path.exists() and yaml.safe_load(path.read_text()) != raw:
        raise RuntimeError(f'Existing config differs: {path}')
    path.write_text(yaml.safe_dump(raw, sort_keys=False))
    print('PREPARED', path, 'start', start, 'count', count, flush=True)
(roots[0].parent / '1028-wino-remaining-boundary.txt').write_text(str(mid))
print('MISSING_AT_SNAPSHOT', len(missing), 'MID', mid, flush=True)
PY
mid=$(cat "$project/1028-wino-remaining-boundary.txt")
held=("$old_resume" "$smoke" 17088)
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
    if (( idx == 0 )); then start=800; count=$((mid-800)); else start=$mid; count=$((1267-mid)); fi
    args=("$repo/h200/fpmi_wino_shard_worker.sh" "$repo" "$root" "$start" "$count")
    job=$(sbatch --parsable "${common[@]}" --dependency="afterany:$old_job" --job-name="wino-rem-$idx" --output="$root/run.log" "${args[@]}")
    submitted+=("$job"); printf '%s\n' "$job" > "$root/job-id.txt"
    printf '%q ' sbatch "${common[@]}" --dependency="afterany:$old_job" --output="$root/run.log" "${args[@]}" > "$root/cmd.txt"
    printf '\n' >> "$root/cmd.txt"
    resume=$(sbatch --parsable "${common[@]}" --dependency="afterany:$job" --job-name="wino-rem-resume-$idx" --output="$root/resume.log" "${args[@]}")
    submitted+=("$resume"); resumes+=("$resume"); printf '%s\n' "$resume" > "$root/resume-job.txt"
done
dependency="afterany:${resumes[0]}:${resumes[1]}"
scontrol update "JobId=$smoke" "Dependency=$dependency"
scontrol update "JobId=17088" "Dependency=$dependency"
scancel "$old_resume"
held=("$smoke" 17088)
for job in "${held[@]}"; do scontrol release "$job"; done
held=()
scancel "$old_job"
trap - EXIT
printf 'NEW_A=%s RESUME_A=%s NEW_B=%s RESUME_B=%s MID=%s\n' "${submitted[0]}" "${resumes[0]}" "${submitted[2]}" "${resumes[1]}" "$mid"
squeue -u "$USER" -o '%.8i %.20j %.9T %.10M %.25R'
