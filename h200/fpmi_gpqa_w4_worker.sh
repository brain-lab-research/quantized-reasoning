#!/usr/bin/env bash
# Full GPQA-Diamond, one request at a time; temporary W4 weights live in RAM.
set -euo pipefail
repo=$1
root=$2
python="/home/data/$USER/envs/low-bit-vllm/bin/python"
export PYTHONPATH="$repo" PATH="$(dirname "$python"):$PATH" HF_HOME="/home/data/$USER/hf-cache"
export WANDB_MODE=offline PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 VLLM_WORKER_MULTIPROC_METHOD=spawn
unset ROCR_VISIBLE_DEVICES
cd "$repo"
started=$SECONDS
set -a; . "$HOME/.s3-cod.env"; set +a
sync_once() { "$python" "$repo/h200/sync.py" --root "$root" --prefix "shveikinda/h200/low-bit/$(basename "$root")"; }
(while sleep 180; do sync_once || true; done) & sync_pid=$!
stage=''
finish() {
    rc=$?; trap - EXIT
    kill "$sync_pid" 2>/dev/null || true
    sync_once || rc=1
    if [[ -n "$stage" ]]; then
        case "$stage" in /dev/shm/lowbit-gpqa-w4-"$SLURM_JOB_ID"-*) rm -rf -- "$stage" ;; *) echo INVALID_STAGE >&2; rc=1 ;; esac
    fi
    if (( rc != 0 && SECONDS - started < 2400 )) && [[ -f "$root/job-ids.txt" ]]; then
        while read -r next; do [[ "$next" == "$SLURM_JOB_ID" ]] || scancel "$next" 2>/dev/null || true; done < "$root/job-ids.txt"
    fi
    exit "$rc"
}
trap finish EXIT
trap 'exit 143' TERM INT
if [[ -f "$root/artifacts/summary.json" ]] && "$python" - "$root/artifacts/summary.json" <<'PY'
import json, sys
d=json.load(open(sys.argv[1])); sys.exit(0 if d.get('complete') and d.get('completed') == 198 else 1)
PY
then
    echo GPQA_W4_ALREADY_COMPLETE
    exit 0
fi
available=$(df -Pk /dev/shm | awk 'NR==2 {print $4}')
(( available >= 22000000 )) || { echo "TMPFS_SPACE_INSUFFICIENT available_kib=$available" >&2; exit 1; }
stage=$(mktemp -d "/dev/shm/lowbit-gpqa-w4-${SLURM_JOB_ID:?}-XXXXXXXX")
printf 'GPU_JOB=%s DEVICE=%s\n' "$SLURM_JOB_ID" "$CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=uuid,memory.total --format=csv
"$python" - "$stage/model" "$root/model-revision.json" <<'PY'
import json, sys
from pathlib import Path
from huggingface_hub import HfApi, snapshot_download
model = 'kaitchup/Qwen3-32B-autoround-4bit-gptq'
record = Path(sys.argv[2])
revision = json.loads(record.read_text())['revision'] if record.exists() else HfApi().model_info(model).sha
record.write_text(json.dumps({'model_id': model, 'revision': revision}, indent=2))
print('DOWNLOAD_START W4', flush=True)
snapshot_download(model, revision=revision, local_dir=sys.argv[1], max_workers=4,
    allow_patterns=['*.json', '*.safetensors', '*.txt', '*.jinja', '*.model'])
print('DOWNLOAD_DONE W4', flush=True)
PY
if [[ ! -f "$root/smoke/summary.json" ]]; then
    "$python" scripts/profile_baseline_latency.py --config "$root/cfg.yaml" --output "$root/smoke" --examples 1 --model-path "$stage/model"
fi
"$python" - "$root/smoke/summary.json" <<'PY'
import json, sys
d=json.load(open(sys.argv[1])); assert d.get('completed') == 1 and d.get('total_output_tokens', 0) > 0, d
print('GPQA_W4_SMOKE_OK', flush=True)
PY
"$python" scripts/profile_baseline_latency.py --config "$root/cfg.yaml" --output "$root/artifacts" --examples 198 --model-path "$stage/model"
echo GPQA_W4_PROFILE_COMPLETE
