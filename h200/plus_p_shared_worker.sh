#!/usr/bin/env bash
set -euo pipefail
repo=$1; root=$2; owner=$3
python="/home/data/$USER/envs/low-bit-vllm/bin/python"
export PYTHONPATH="$repo" PATH="$(dirname "$python"):$PATH" HF_HOME="/home/data/$USER/hf-cache"
export WANDB_MODE=offline PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 VLLM_WORKER_MULTIPROC_METHOD=spawn
unset ROCR_VISIBLE_DEVICES
set -a; . "$HOME/.s3-cod.env"; set +a
started=$SECONDS
stage=$(mktemp -d "/dev/shm/lowbit-plus-p-${SLURM_JOB_ID:?}-XXXXXXXX")
sync_once() { "$python" "$repo/h200/sync.py" --root "$root/$owner" --prefix "shveikinda/h200/low-bit/1030-plus-p-shared-backup/$owner"; }
(while sleep 180; do sync_once || echo BACKUP_RETRY; done) & sync_pid=$!
finish() {
    rc=$?; trap - EXIT; kill "$sync_pid" 2>/dev/null || true; sync_once || rc=1
    case "$stage" in /dev/shm/lowbit-plus-p-"$SLURM_JOB_ID"-*) rm -rf -- "$stage" ;; *) rc=1 ;; esac
    ids="$root/jobs-$owner.txt"
    if ((rc != 0 && SECONDS-started < 2400)) && [[ -f "$ids" ]]; then
        while read -r job; do [[ "$job" == "$SLURM_JOB_ID" ]] || scancel "$job" 2>/dev/null || true; done < "$ids"
    fi
    exit "$rc"
}
trap finish EXIT
trap 'exit 143' TERM INT
mkdir -p "$root/$owner"
"$python" "$repo/scripts/prepare_plus_p_manifest.py" --root "$root/$owner"
"$python" - "$stage/model" <<'PY'
import hashlib,sys,boto3
from pathlib import Path
model=Path(sys.argv[1]); model.mkdir(parents=True,exist_ok=True)
s=boto3.client('s3',endpoint_url='https://s3.cod.phystech.edu'); prefix='shveikinda/h200/low-bit/1029-distill-w2-temporary-model/'
for name in ['config.json','generation_config.json','model.safetensors','quantization_config.json','special_tokens_map.json','tokenizer.json','tokenizer_config.json']:
    s.download_file('iii-storage2',prefix+name,str(model/name))
h=hashlib.sha256()
with (model/'model.safetensors').open('rb') as f:
    for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
if h.hexdigest()!='c8c207ac49b286fb8fa9d5708878da50b33b4e6c4e20105b0d735f0adaea273f':raise ValueError('MODEL_SHA_MISMATCH')
print('MODEL_SHA_OK',flush=True)
PY
smoke="$root/$owner/smoke.json"
if [[ ! -f "$smoke" ]]; then "$python" "$repo/h200/smoke_shared_distill_w2.py" --model-path "$stage/model" --output "$smoke"; fi
"$python" - "$smoke" <<'PY'
import json,sys
s=json.load(open(sys.argv[1])); assert s.get('correct',0)>=2,s
print('W2_SMOKE_OK',flush=True)
PY
"$python" "$repo/scripts/collect_multi_bench_traces.py" --manifest "$root/$owner/plus-p-manifest.json" \
    --model-path "$stage/model" --plans-root "$root/plans" --max-tokens 30720 \
    --max-num-seqs 32 --max-outstanding 64 --kv-cache-gib 80 --memory-utilization 0.75 \
    --reverse --shared-prefix shveikinda/h200/low-bit/1030-plus-p-queue \
    --shared-owner "$owner" --shared-fingerprint plus-p-c8c207ac-21e432748b35b06a5df3fd2d
echo WORKER_COMPLETE
