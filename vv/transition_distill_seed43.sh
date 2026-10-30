#!/usr/bin/env bash
# Keep GPU 7 occupied by the old job until the new seeded engine passes smoke.
set -euo pipefail
repo=$1
old="$HOME/projects/low-bit-reasoning/exps/1029-r1-distill-llama-8b-w2-full"
root="$HOME/projects/low-bit-reasoning/exps/1030-distill-w2-seed43"
model="$HOME/projects/low-bit-reasoning/exps/1029-r1-distill-llama-8b-autoround-256/model-gptq2-g32/model-w2g32"
compat="$HOME/projects/low-bit-reasoning/exps/1029-gpqa-rescue-vv/compat"
python=/data/users/shveikinda/miniconda3/bin/python
mkdir -p "$root"
tmux has-session -t =lowbit-r1-multi 2>/dev/null || { echo OLD_QUEUE_NOT_RUNNING; exit 1; }
if tmux has-session -t =lowbit-r1-seed43 2>/dev/null; then echo NEW_QUEUE_ALREADY_RUNNING; exit 1; fi
old_pid=$(pgrep -u "$USER" -f "collect_multi_bench_traces.py --manifest $old/multi-manifest.json" || true)
[[ "$old_pid" =~ ^[0-9]+$ ]] || { echo OLD_PROCESS_NOT_UNIQUE >&2; exit 1; }
export PYTHONPATH="$compat:$repo" CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=GPU-81dfcc32-ea97-c0f1-778e-ff44c4992c1a
export WANDB_MODE=offline PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false VLLM_WORKER_MULTIPROC_METHOD=spawn
"$python" "$repo/scripts/prepare_distill_second_seed.py" --root "$root" --seed 43
barrier="$root/.handoff-ready-$$"
[[ ! -e "$barrier" ]] || { echo STALE_BARRIER >&2; exit 1; }
cat > "$root/cmd.txt" <<EOF
cd $repo
export PYTHONPATH=$PYTHONPATH CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES WANDB_MODE=offline PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false VLLM_WORKER_MULTIPROC_METHOD=spawn
exec $python $repo/scripts/collect_multi_bench_traces.py --manifest $root/manifest.json --model-path $model --max-num-seqs 16 --max-outstanding 32 --kv-cache-gib 40 --memory-utilization 0.33 --local-ready-barrier $barrier
EOF
free=$(nvidia-smi -i 7 --query-gpu=memory.free --format=csv,noheader,nounits)
((free > 50000)) || { echo "INSUFFICIENT_GPU7_FREE_MIB $free" >&2; exit 1; }
tmux new-session -d -s lowbit-r1-seed43 "bash $root/cmd.txt > $root/run.log 2>&1"
for _ in $(seq 1 180); do
    grep -q '^ENGINE_READY' "$root/run.log" 2>/dev/null && break
    tmux has-session -t =lowbit-r1-seed43 2>/dev/null || { echo NEW_ENGINE_FAILED; tail -n 30 "$root/run.log"; exit 1; }
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
touch "$barrier"
echo SEED43_HANDOFF_COMPLETE
