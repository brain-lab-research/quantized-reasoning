#!/usr/bin/env bash
set -euo pipefail
repo=$1; root=$2; owner=$3
python="/home/data/$USER/envs/low-bit-vllm/bin/python"
export PYTHONPATH="$repo" PATH="$(dirname "$python"):$PATH" HF_HOME="/home/data/$USER/hf-cache"
export WANDB_MODE=offline PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 VLLM_WORKER_MULTIPROC_METHOD=spawn
unset ROCR_VISIBLE_DEVICES
set -a; . "$HOME/.s3-cod.env"; set +a
stage=$(mktemp -d "/dev/shm/lowbit-shared-w2-${SLURM_JOB_ID:?}-XXXXXXXX")
sync_once() { "$python" "$repo/h200/sync.py" --root "$root/$owner" --prefix "shveikinda/h200/low-bit/1029-distill-shared-backup/$owner"; }
(while sleep 180; do sync_once || echo BACKUP_RETRY; done) & sync_pid=$!
finish() { rc=$?; trap - EXIT; kill "$sync_pid" 2>/dev/null || true; sync_once || rc=1; case "$stage" in /dev/shm/lowbit-shared-w2-"$SLURM_JOB_ID"-*) rm -rf -- "$stage" ;; *) rc=1 ;; esac; exit "$rc"; }
trap finish EXIT
trap 'exit 143' TERM INT
mkdir -p "$root/$owner"
"$python" "$repo/scripts/prepare_shared_manifest.py" --root "$root/$owner"
"$python" - "$stage/model" <<'PY'
import hashlib,sys,boto3
from pathlib import Path
model=Path(sys.argv[1]); model.mkdir(parents=True,exist_ok=True)
s=boto3.client('s3',endpoint_url='https://s3.cod.phystech.edu'); bucket='iii-storage2'; prefix='shveikinda/h200/low-bit/1029-distill-w2-temporary-model/'
names=['config.json','generation_config.json','model.safetensors','quantization_config.json','special_tokens_map.json','tokenizer.json','tokenizer_config.json']
for name in names:
    print('DOWNLOAD',name,flush=True);s.download_file(bucket,prefix+name,str(model/name))
h=hashlib.sha256()
with (model/'model.safetensors').open('rb') as f:
    for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
if h.hexdigest()!='c8c207ac49b286fb8fa9d5708878da50b33b4e6c4e20105b0d735f0adaea273f':raise ValueError('MODEL_SHA_MISMATCH')
print('MODEL_SHA_OK',flush=True)
PY
"$python" "$repo/scripts/collect_multi_bench_traces.py" --manifest "$root/$owner/shared-manifest.json" \
    --model-path "$stage/model" --max-num-seqs 16 --max-outstanding 32 --kv-cache-gib 80 \
    --memory-utilization 0.75 --reverse --shared-prefix shveikinda/h200/low-bit/1029-distill-shared \
    --shared-owner "$owner" --shared-fingerprint distill-w2-c8c207ac-seed42-native-v2
echo SHARED_WORKER_COMPLETE
