#!/usr/bin/env bash
set -euo pipefail
repo=$1; root=$2; worker=$3
python="/home/data/$USER/envs/low-bit-vllm/bin/python"
export PYTHONPATH="$repo" PATH="$(dirname "$python"):$PATH" HF_HOME="/home/data/$USER/hf-cache"
export WANDB_MODE=offline PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 VLLM_WORKER_MULTIPROC_METHOD=spawn
unset ROCR_VISIBLE_DEVICES
cd "$repo"
started=$SECONDS
stage=$(mktemp -d "/dev/shm/lowbit-distill-${SLURM_JOB_ID:?}-XXXXXXXX")
set -a; . "$HOME/.s3-cod.env"; set +a
sync_once() { "$python" "$repo/h200/sync.py" --root "$root" --prefix "shveikinda/h200/low-bit/$(basename "$root")"; }
(while sleep 180; do sync_once || echo BACKUP_RETRY; done) & sync_pid=$!
finish() {
    rc=$?; trap - EXIT
    kill "$sync_pid" 2>/dev/null || true
    sync_once || rc=1
    case "$stage" in /dev/shm/lowbit-distill-"$SLURM_JOB_ID"-*) rm -rf -- "$stage" ;; *) rc=1 ;; esac
    ids="$root/jobs-$worker.txt"
    if ((rc != 0 && SECONDS-started < 2400)) && [[ -f "$ids" ]]; then
        while read -r job; do [[ "$job" == "$SLURM_JOB_ID" ]] || scancel "$job" 2>/dev/null || true; done < "$ids"
    fi
    exit "$rc"
}
trap finish EXIT
trap 'exit 143' TERM INT
echo "START job=$SLURM_JOB_ID worker=$worker device=$CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=uuid,memory.total --format=csv
"$python" - "$stage/model" <<'PY'
import sys
from huggingface_hub import snapshot_download
revision='6a6f4aa4197940add57724a7707d069478df56b1'
print('DOWNLOAD_START',revision,flush=True)
snapshot_download('deepseek-ai/DeepSeek-R1-Distill-Llama-8B',revision=revision,
                  local_dir=sys.argv[1],max_workers=4,
                  allow_patterns=['*.json','*.safetensors','*.txt','*.jinja','*.model'])
print('DOWNLOAD_DONE',flush=True)
PY
smoke="$root/smoke-$worker"
if [[ ! -f "$smoke/run-id.txt" ]]; then
    "$python" "$repo/scripts/collect_baseline_traces.py" "$root/1029-r1-distill-llama-8b-strategyqa-fp16/cfg.yaml" \
        --output "$smoke" --max-examples 2 --max-tokens 128 --max-num-seqs 2 --kv-cache-gib 8 \
        --memory-utilization 0.5 --native-chat --model-path "$stage/model"
fi
"$python" - "$smoke" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1]); run=(p/'run-id.txt').read_text().strip(); s=json.loads((p/run/'summary.json').read_text())
assert s.get('n_judged')==2 and s.get('n_errors')==0 and s.get('generated_tokens',0)>0,s
print('SMOKE_OK',flush=True)
PY
"$python" "$repo/vv/distill_llama_matrix.py" --root "$root" --model-path "$stage/model" --precision fp16 --shared
echo DISTILL_WORKER_COMPLETE
