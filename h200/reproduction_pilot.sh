#!/usr/bin/env bash
# Prepare or run two controlled MATH-500 decoding pilots on FPMI.
set -euo pipefail
repo=$(cd "$(dirname "$0")/.." && pwd)
root="/home/data/$USER/projects/low-bit-reasoning/exps/1028-reproduction-pilot"
python="/home/data/$USER/envs/low-bit-vllm/bin/python"
export PYTHONPATH="$repo" HF_HOME="/home/data/$USER/hf-cache"
export WANDB_MODE=offline PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 VLLM_WORKER_MULTIPROC_METHOD=spawn
unset ROCR_VISIBLE_DEVICES

if [[ ${1:-} == smoke ]]; then
    export HF_HUB_OFFLINE=1
    cd "$repo"
    "$python" scripts/collect_baseline_traces.py "$root/greedy/cfg.yaml" --output "$root/smoke/artifacts" \
        --max-examples 2 --max-tokens 512 --max-num-seqs 2 --kv-cache-gib 8 \
        --memory-utilization 0.8 --open-thinking
    "$python" - "$root/smoke/artifacts" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
summary = json.loads((root / (root / 'run-id.txt').read_text().strip() / 'summary.json').read_text())
assert summary['n_judged'] == 2 and summary['n_errors'] == 0, summary
print('PILOT_SMOKE_OK', flush=True)
PY
    exit 0
fi

if [[ ${1:-} == worker ]]; then
    started=$SECONDS
    export HF_HUB_OFFLINE=1
    set -a; . "$HOME/.s3-cod.env"; set +a
    sync_once() { "$python" "$repo/h200/sync.py" --root "$root" --prefix "shveikinda/h200/low-bit/$(basename "$root")"; }
    (while sleep 180; do sync_once || true; done) & sync_pid=$!
    finish() {
        rc=$?; trap - EXIT
        kill "$sync_pid" 2>/dev/null || true
        sync_once || rc=1
        if (( rc != 0 && SECONDS - started < 2400 )) && [[ -f "$root/resume-job.txt" ]]; then
            next=$(cat "$root/resume-job.txt")
            [[ "$next" == "${SLURM_JOB_ID:-}" ]] || scancel "$next" 2>/dev/null || true
        fi
        exit "$rc"
    }
    trap finish EXIT
    trap 'exit 143' TERM INT
    cd "$repo"
    for name in greedy unrestricted; do
        echo "PILOT_START $name"
        "$python" scripts/collect_baseline_traces.py "$root/$name/cfg.yaml" --output "$root/$name/artifacts" \
            --max-examples 100 --max-tokens 32768 --max-num-seqs 32 --kv-cache-gib 64 \
            --memory-utilization 0.8 --open-thinking
        echo "PILOT_DONE $name"
    done
    exit 0
fi

if [[ ${1:-} != prepare ]]; then
    echo 'Usage: reproduction_pilot.sh prepare | worker' >&2
    exit 2
fi
mkdir -p "$root/greedy" "$root/unrestricted"
"$python" - "$repo" "$root" <<'PY'
import copy
import sys
from pathlib import Path
import yaml
from quantlab.config.schema import ExperimentConfig
repo, root = map(Path, sys.argv[1:])
base = yaml.safe_load((repo / 'configs/math500/qwen32b_gptq2bit/single_gptq2bit.yaml').read_text())
for name, temperature in (('greedy', 0.0), ('unrestricted', 0.6)):
    cfg = copy.deepcopy(base)
    cfg['experiment_name'] = f'1028-qwen32b-w2-math500-{name}'
    cfg['benchmark'].update(max_examples=100, seed=42, start_index=0)
    actor = cfg['actors'][0]
    actor['generation_params'].update(max_new_tokens=32768, temperature=temperature, top_p=1.0, top_k=-1, seed=42)
    actor['backend_kwargs'].pop('cuda_visible_devices', None)
    actor['backend_kwargs'].update(max_model_len=36864, enforce_eager=False, enable_chunked_prefill=True)
    cfg['pipeline'][0]['max_new_tokens'] = 32768
    cfg['pipeline_max_total_tokens'] = 32768
    cfg['output']['base_dir'] = str(root / name / 'artifacts')
    cfg['wandb'] = {'enabled': False}
    ExperimentConfig.model_validate(cfg)
    path = root / name / 'cfg.yaml'
    if path.exists() and yaml.safe_load(path.read_text()) != cfg:
        raise RuntimeError(f'Existing config differs: {path}')
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    print('PREPARED', path)
PY
