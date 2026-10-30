#!/usr/bin/env bash
# Keep the current GPU engine running until its larger replacement passes smoke.
set -euo pipefail
repo=$1
root="$HOME/projects/low-bit-reasoning/exps/1030-distill-w2-seed43"
model="$HOME/projects/low-bit-reasoning/exps/1029-r1-distill-llama-8b-autoround-256/model-gptq2-g32/model-w2g32"
compat="$HOME/projects/low-bit-reasoning/exps/1029-gpqa-rescue-vv/compat"
python=/data/users/shveikinda/miniconda3/bin/python
tmux has-session -t =lowbit-r1-seed43 2>/dev/null || { echo OLD_QUEUE_NOT_RUNNING; exit 1; }
if tmux has-session -t =lowbit-r1-seed43-kv80 2>/dev/null; then echo NEW_QUEUE_ALREADY_RUNNING; exit 1; fi
old_pid=$(pgrep -u "$USER" -f "collect_multi_bench_traces.py --manifest $root/manifest.json" || true)
[[ "$old_pid" =~ ^[0-9]+$ ]] || { echo OLD_PROCESS_NOT_UNIQUE >&2; exit 1; }
expected=c8c207ac49b286fb8fa9d5708878da50b33b4e6c4e20105b0d735f0adaea273f
actual=$(sha256sum "$model/model.safetensors" | cut -d ' ' -f 1)
[[ "$actual" == "$expected" ]] || { echo MODEL_SHA_MISMATCH >&2; exit 1; }
barrier="$root/.kv80-handoff-ready-$$"
[[ ! -e "$barrier" ]] || { echo STALE_BARRIER >&2; exit 1; }
export PYTHONPATH="$compat:$repo" CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=GPU-81dfcc32-ea97-c0f1-778e-ff44c4992c1a
export WANDB_MODE=offline PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false VLLM_WORKER_MULTIPROC_METHOD=spawn
cat > "$root/cmd-kv80.txt" <<EOF
cd $repo
export PYTHONPATH=$PYTHONPATH CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES WANDB_MODE=offline PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false VLLM_WORKER_MULTIPROC_METHOD=spawn
exec $python $repo/scripts/collect_multi_bench_traces.py --manifest $root/manifest.json --model-path $model --max-num-seqs 32 --max-outstanding 64 --kv-cache-gib 80 --memory-utilization 0.60 --allow-scheduler-change --local-ready-barrier $barrier
EOF
free=$(nvidia-smi -i 7 --query-gpu=memory.free --format=csv,noheader,nounits)
((free > 93000)) || { echo "INSUFFICIENT_GPU7_FREE_MIB $free" >&2; exit 1; }
tmux new-session -d -s lowbit-r1-seed43-kv80 "bash $root/cmd-kv80.txt > $root/run-kv80.log 2>&1"
for _ in $(seq 1 180); do
    grep -q '^ENGINE_READY' "$root/run-kv80.log" 2>/dev/null && break
    tmux has-session -t =lowbit-r1-seed43-kv80 2>/dev/null || { echo NEW_ENGINE_FAILED; tail -n 30 "$root/run-kv80.log"; exit 1; }
    sleep 2
done
grep -q '^ENGINE_READY' "$root/run-kv80.log" || { echo NEW_ENGINE_NOT_READY; exit 1; }
tmux has-session -t =lowbit-r1-seed43-kv80 2>/dev/null || { echo NEW_ENGINE_NO_LONGER_RUNNING; exit 1; }
kill -0 "$old_pid" || { echo OLD_ENGINE_ALREADY_GONE; exit 1; }
echo KV80_ENGINE_READY_OLD_STILL_RUNNING
kill -INT "$old_pid"
for _ in $(seq 1 120); do
    kill -0 "$old_pid" 2>/dev/null || break
    sleep 1
done
if kill -0 "$old_pid" 2>/dev/null; then echo OLD_ENGINE_DID_NOT_EXIT >&2; exit 1; fi
touch "$barrier"
if ! tmux has-session -t =lowbit-r1-seed43-kv80-backup 2>/dev/null; then
    tmux new-session -d -s lowbit-r1-seed43-kv80-backup "bash $repo/vv/sync_distill_seed43.sh $repo lowbit-r1-seed43-kv80 > $root/backup-kv80.log 2>&1"
fi
echo KV80_HANDOFF_COMPLETE
