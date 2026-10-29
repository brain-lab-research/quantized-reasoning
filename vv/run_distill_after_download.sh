#!/usr/bin/env bash
# Switch the temporary single-request profiler to full FP16 traces after weights arrive.
set -euo pipefail
repo=$1
root=$2
python=/data/users/shveikinda/miniconda3/bin/python
compat="$HOME/projects/low-bit-reasoning/exps/1029-gpqa-rescue-vv/compat"
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=GPU-81dfcc32-ea97-c0f1-778e-ff44c4992c1a
export HF_HOME=/data/hf_cache PYTHONPATH="$compat:$repo" WANDB_MODE=offline PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false VLLM_WORKER_MULTIPROC_METHOD=spawn
cd "$repo"
while [[ ! -f "$root/download.done" ]]; do
    if ! tmux has-session -t lowbit-r1-download 2>/dev/null; then
        echo DOWNLOAD_FAILED; exit 1
    fi
    sleep 15
done
"$python" "$repo/vv/distill_llama_matrix.py" --root "$root" --prepare
if tmux has-session -t lowbit-gpqa-w2-latency 2>/dev/null; then
    echo STOPPING_TEMPORARY_W2_PROFILE
    tmux send-keys -t lowbit-gpqa-w2-latency C-c
fi
for _ in $(seq 1 60); do
    free=$(nvidia-smi -i 7 --query-gpu=memory.free --format=csv,noheader,nounits)
    (( free >= 125000 )) && break
    sleep 2
done
(( free >= 125000 )) || { echo GPU7_NOT_FREE "$free"; exit 1; }
smoke_cfg="$root/1029-r1-distill-llama-8b-strategyqa-fp16/cfg.yaml"
"$python" "$repo/scripts/collect_baseline_traces.py" "$smoke_cfg" --output "$root/smoke" \
    --max-examples 2 --max-tokens 128 --max-num-seqs 2 --kv-cache-gib 8 --memory-utilization 0.5 \
    --native-chat --model-path "$root/model"
"$python" - "$root/smoke" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1]); run = (root / 'run-id.txt').read_text().strip()
summary = json.loads((root / run / 'summary.json').read_text())
assert summary.get('n_judged') == 2 and summary.get('n_errors') == 0, summary
assert summary.get('generated_tokens', 0) > 0, summary
print('SMOKE_OK', flush=True)
PY
"$python" "$repo/vv/distill_llama_matrix.py" --root "$root" --model-path "$root/model"
echo FP16_MATRIX_COMPLETE
