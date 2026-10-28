#!/usr/bin/env bash
# One-GPU, batch-1 online GPQA rescue; FP16 weights are staged in tmpfs.
set -euo pipefail
repo=$1
root=$2
python="/home/data/$USER/envs/low-bit-vllm/bin/python"
w2="/home/data/$USER/hf-cache/hub/models--kaitchup--Qwen3-32B-autoround-2bit-gptq/snapshots/39139ba48db8ee12088315c9fab08cf7b1b1c3ba"
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
        case "$stage" in /dev/shm/lowbit-gpqa-rescue-"$SLURM_JOB_ID"-*) rm -rf -- "$stage" ;; *) echo INVALID_STAGE >&2; rc=1 ;; esac
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
    echo GPQA_RESCUE_ALREADY_COMPLETE
    exit 0
fi
[[ -f "$w2/config.json" && -f "$w2/tokenizer.json" ]] || { echo W2_SNAPSHOT_MISSING >&2; exit 1; }
available=$(df -Pk /dev/shm | awk 'NR==2 {print $4}')
(( available >= 70000000 )) || { echo "TMPFS_SPACE_INSUFFICIENT available_kib=$available" >&2; exit 1; }
stage=$(mktemp -d "/dev/shm/lowbit-gpqa-rescue-${SLURM_JOB_ID:?}-XXXXXXXX")
printf 'GPU_JOB=%s DEVICE=%s\n' "$SLURM_JOB_ID" "$CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=uuid,memory.total --format=csv
"$python" - "$stage/model" "$root/fp16-revision.json" <<'PY'
import json, sys
from pathlib import Path
from huggingface_hub import HfApi, snapshot_download
model='Qwen/Qwen3-32B'
record=Path(sys.argv[2])
revision=json.loads(record.read_text())['revision'] if record.exists() else HfApi().model_info(model).sha
record.write_text(json.dumps({'model_id':model, 'revision':revision}, indent=2))
print('DOWNLOAD_START FP16', revision, flush=True)
snapshot_download(model, revision=revision, local_dir=sys.argv[1], max_workers=4,
                  allow_patterns=['*.json', '*.safetensors', '*.txt', '*.jinja', '*.model'])
print('DOWNLOAD_DONE FP16', flush=True)
PY
revision=$("$python" -c 'import json,sys;print(json.load(open(sys.argv[1]))["revision"])' "$root/fp16-revision.json")
common=(--config "$root/cfg.yaml" --w2-path "$w2" --fp16-path "$stage/model" --fp16-revision "$revision")
if [[ ! -f "$root/smoke/summary.json" ]]; then
    "$python" scripts/measure_gpqa_rescue.py "${common[@]}" --output "$root/smoke" --examples 1 --smoke
fi
"$python" - "$root/smoke/summary.json" <<'PY'
import json, sys
d=json.load(open(sys.argv[1])); assert d.get('completed') == 1 and d.get('w2_output_tokens',0) > 0, d
print('GPQA_RESCUE_SMOKE_OK', flush=True)
PY
"$python" scripts/measure_gpqa_rescue.py "${common[@]}" --output "$root/artifacts" --examples 198
echo GPQA_RESCUE_PROFILE_COMPLETE
