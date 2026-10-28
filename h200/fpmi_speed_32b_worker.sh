#!/usr/bin/env bash
# Profile FP16/W4/W2 on the same GPU, staging the two uncached models in tmpfs.
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
stage=$(mktemp -d "/dev/shm/lowbit-speed-${SLURM_JOB_ID:?}-XXXXXXXX")
set -a; . "$HOME/.s3-cod.env"; set +a
sync_once() { "$python" "$repo/h200/sync.py" --root "$root" --prefix "shveikinda/h200/low-bit/$(basename "$root")"; }
(while sleep 180; do sync_once || true; done) & sync_pid=$!
finish() {
    rc=$?; trap - EXIT
    kill "$sync_pid" 2>/dev/null || true
    sync_once || rc=1
    case "$stage" in /dev/shm/lowbit-speed-"$SLURM_JOB_ID"-*) rm -rf -- "$stage" ;; *) echo INVALID_STAGE >&2; rc=1 ;; esac
    if (( rc != 0 && SECONDS - started < 2400 )) && [[ -f "$root/resume-job.txt" ]]; then
        next=$(cat "$root/resume-job.txt")
        [[ "$next" == "$SLURM_JOB_ID" ]] || scancel "$next" 2>/dev/null || true
    fi
    exit "$rc"
}
trap finish EXIT
trap 'exit 143' TERM INT
printf 'GPU_JOB=%s DEVICE=%s\n' "$SLURM_JOB_ID" "$CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=uuid,memory.total --format=csv
for mode in w2 w4 fp16; do
    if [[ -f "$root/$mode/artifacts/summary.json" ]] && "$python" - "$root/$mode/artifacts/summary.json" <<'PY'
import json, sys
d=json.load(open(sys.argv[1])); sys.exit(0 if d.get('complete') and d.get('completed') == 20 else 1)
PY
    then
        echo "PROFILE_ALREADY_COMPLETE $mode"
        continue
    fi
    model_path=()
    if [[ "$mode" != w2 ]]; then
        available=$(df -Pk /dev/shm | awk 'NR==2 {print $4}')
        required_kib=22000000
        [[ "$mode" == fp16 ]] && required_kib=70000000
        (( available >= required_kib )) || { echo "TMPFS_SPACE_INSUFFICIENT mode=$mode available_kib=$available" >&2; exit 1; }
        model_id='kaitchup/Qwen3-32B-autoround-4bit-gptq'
        [[ "$mode" == fp16 ]] && model_id='Qwen/Qwen3-32B'
        model_dir="$stage/$mode"
        "$python" - "$model_id" "$model_dir" <<'PY'
import sys
from huggingface_hub import snapshot_download
print('DOWNLOAD_START', sys.argv[1], flush=True)
snapshot_download(sys.argv[1], local_dir=sys.argv[2], max_workers=4,
    allow_patterns=['*.json', '*.safetensors', '*.txt', '*.jinja', '*.model'])
print('DOWNLOAD_DONE', sys.argv[1], flush=True)
PY
        model_path=(--model-path "$model_dir")
    fi
    echo "SMOKE_START $mode"
    "$python" scripts/profile_baseline_latency.py --config "$root/$mode/cfg.yaml" --output "$root/$mode/smoke" --examples 1 "${model_path[@]}"
    "$python" - "$root/$mode/smoke/summary.json" <<'PY'
import json, sys
d=json.load(open(sys.argv[1])); assert d.get('completed') == 1 and d.get('total_output_tokens', 0) > 0, d
print('SPEED_SMOKE_OK', flush=True)
PY
    echo "PROFILE_START $mode"
    "$python" scripts/profile_baseline_latency.py --config "$root/$mode/cfg.yaml" --output "$root/$mode/artifacts" --examples 20 "${model_path[@]}"
    if [[ "$mode" != w2 ]]; then rm -rf -- "$stage/$mode"; fi
    sync_once
    echo "PROFILE_DONE $mode"
done
