#!/usr/bin/env bash
# The replacement engine is loaded and waiting before the old collector is interrupted.
set -euo pipefail
repo=$1
root="$HOME/projects/low-bit-reasoning/exps/1029-distill-shared-w2-vv"
old="$HOME/projects/low-bit-reasoning/exps/1029-r1-distill-llama-8b-w2-full"
pilot="$HOME/projects/low-bit-reasoning/exps/1029-r1-distill-llama-8b-autoround-256/compare"
model="$HOME/projects/low-bit-reasoning/exps/1029-r1-distill-llama-8b-autoround-256/model-gptq2-g32/model-w2g32"
compat="$HOME/projects/low-bit-reasoning/exps/1029-gpqa-rescue-vv/compat"
deps="$HOME/projects/low-bit-reasoning/exps/1029-r1-distill-llama-8b/s3-pydeps"
python=/data/users/shveikinda/miniconda3/bin/python
export PYTHONPATH="$compat:$repo:$deps" WANDB_MODE=offline PYTHONUNBUFFERED=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=GPU-81dfcc32-ea97-c0f1-778e-ff44c4992c1a
export TOKENIZERS_PARALLELISM=false VLLM_WORKER_MULTIPROC_METHOD=spawn
set -a; . "$HOME/.s3-cod.env"; set +a
prefix=shveikinda/h200/low-bit/1029-distill-shared
fingerprint=distill-w2-c8c207ac-seed42-native-v2
sha=c8c207ac49b286fb8fa9d5708878da50b33b4e6c4e20105b0d735f0adaea273f
mkdir -p "$root"
if [[ "${2:-}" == '--wait-request' ]]; then
    "$python" - "$prefix/switch-request.json" "$fingerprint" <<'PY'
import json,sys,time,boto3
from botocore.exceptions import ClientError
s=boto3.client('s3',endpoint_url='https://s3.cod.phystech.edu')
while True:
    try:
        obj=s.get_object(Bucket='iii-storage2',Key=sys.argv[1]); data=json.loads(obj['Body'].read())
        if data.get('fingerprint')!=sys.argv[2]:raise ValueError('STALE_SWITCH_REQUEST')
        print('W4_COMPLETED_HANDOFF_REQUEST',flush=True);break
    except ClientError as exc:
        if exc.response.get('Error',{}).get('Code') not in ('404','NoSuchKey','NotFound'):raise
        time.sleep(20)
PY
fi
if tmux has-session -t =lowbit-r1-shared 2>/dev/null; then echo SHARED_ALREADY_RUNNING; exit 1; fi
tmux has-session -t =lowbit-r1-multi 2>/dev/null || { echo OLD_QUEUE_NOT_RUNNING; exit 1; }
old_pid=$(pgrep -u "$USER" -f "collect_multi_bench_traces.py --manifest $old/multi-manifest.json" || true)
[[ "$old_pid" =~ ^[0-9]+$ ]] || { echo OLD_PROCESS_NOT_UNIQUE >&2; exit 1; }
"$python" "$repo/scripts/prepare_shared_manifest.py" --root "$root"
seed=("$python" "$repo/scripts/seed_shared_queue.py" --source "$pilot" --source "$old" --model-path "$model" --model-sha256 "$sha" --prefix "$prefix" --fingerprint "$fingerprint")
"${seed[@]}"
cat > "$root/cmd.txt" <<EOF
cd $repo
export PYTHONPATH=$PYTHONPATH WANDB_MODE=offline PYTHONUNBUFFERED=1 CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES TOKENIZERS_PARALLELISM=false VLLM_WORKER_MULTIPROC_METHOD=spawn
set -a; . \$HOME/.s3-cod.env; set +a
exec $python $repo/scripts/collect_multi_bench_traces.py --manifest $root/shared-manifest.json --model-path $model --max-num-seqs 16 --max-outstanding 32 --kv-cache-gib 32 --memory-utilization 0.30 --shared-prefix $prefix --shared-owner vv --shared-fingerprint $fingerprint --ready-barrier $prefix/ready.json
EOF
free=$(nvidia-smi -i 7 --query-gpu=memory.free --format=csv,noheader,nounits)
(( free > 49000 )) || { echo "INSUFFICIENT_GPU7_FREE_MIB $free" >&2; exit 1; }
tmux new-session -d -s lowbit-r1-shared "bash $root/cmd.txt > $root/run.log 2>&1"
for _ in $(seq 1 180); do
    grep -q '^ENGINE_READY' "$root/run.log" 2>/dev/null && break
    tmux has-session -t =lowbit-r1-shared 2>/dev/null || { echo NEW_ENGINE_FAILED; tail -n 40 "$root/run.log"; exit 1; }
    sleep 2
done
grep -q '^ENGINE_READY' "$root/run.log" || { echo NEW_ENGINE_NOT_READY; exit 1; }
kill -0 "$old_pid" || { echo OLD_ENGINE_ALREADY_GONE; exit 1; }
echo NEW_ENGINE_READY_OLD_STILL_RUNNING
kill -INT "$old_pid"
for _ in $(seq 1 120); do
    kill -0 "$old_pid" 2>/dev/null || break
    sleep 1
done
if kill -0 "$old_pid" 2>/dev/null; then echo OLD_ENGINE_DID_NOT_EXIT >&2; exit 1; fi
echo OLD_ENGINE_EXITED
"${seed[@]}" --ready-key "$prefix/ready.json"
echo HANDOFF_COMPLETE
