#!/usr/bin/env bash
set -euo pipefail
repo=$1; root=$2
mode=${3:-after-fp16}
python=/data/users/shveikinda/miniconda3/bin/python
compat="$HOME/projects/low-bit-reasoning/exps/1029-gpqa-rescue-vv/compat"
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=GPU-81dfcc32-ea97-c0f1-778e-ff44c4992c1a
export HF_HOME=/data/hf_cache PYTHONPATH="$compat:$repo" WANDB_MODE=offline PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false VLLM_WORKER_MULTIPROC_METHOD=spawn
cd "$repo"
if [[ "$mode" == after-fp16 ]]; then
    while tmux has-session -t lowbit-r1-fp16 2>/dev/null; do sleep 30; done
    grep -q '^FP16_MATRIX_COMPLETE$' "$root/run.log" || { echo FP16_FAILED_INSPECT_LOG; exit 1; }
elif [[ "$mode" != first ]]; then
    echo INVALID_MODE >&2; exit 1
fi
[[ -f "$root/calibration/c4-windows.jsonl" && -f "$root/training_candidates.jsonl" ]] || { echo MISSING_CALIBRATION; exit 1; }
free=$(nvidia-smi -i 7 --query-gpu=memory.free --format=csv,noheader,nounits)
(( free >= 125000 )) || { echo GPU7_NOT_FREE "$free"; exit 1; }
if [[ ! -f "$root/calibration/teacher-windows.jsonl" ]]; then
    "$python" "$repo/vv/prepare_distill_teacher.py" --root "$root" --count 64
fi
if [[ ! -f "$root/model-gptq2-g32/calibration-provenance.json" ]]; then
    PYTHONPATH="$root/gptq-pydeps:$repo" "$python" "$repo/vv/quantize_distill_gptq2.py" --root "$root"
fi
"$python" "$repo/vv/smoke_distill_w2_quality.py" --root "$root"
"$python" "$repo/vv/distill_llama_matrix.py" --root "$root" --prepare --precision w2
cfg="$root/1029-r1-distill-llama-8b-strategyqa-w2/cfg.yaml"
"$python" "$repo/scripts/collect_baseline_traces.py" "$cfg" --output "$root/smoke-w2" \
    --max-examples 2 --max-tokens 128 --max-num-seqs 2 --kv-cache-gib 8 --memory-utilization 0.5 \
    --native-chat --model-path "$root/model-gptq2-g32"
"$python" - "$root/smoke-w2" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1]); run = (root / 'run-id.txt').read_text().strip()
summary = json.loads((root / run / 'summary.json').read_text())
assert summary.get('n_judged') == 2 and summary.get('n_errors') == 0, summary
assert summary.get('generated_tokens', 0) > 0, summary
print('W2_SMOKE_OK', flush=True)
PY
"$python" "$repo/vv/distill_llama_matrix.py" --root "$root" --precision w2 --model-path "$root/model-gptq2-g32"
echo W2_MATRIX_COMPLETE
if [[ "$mode" == first ]]; then
    "$python" "$repo/vv/distill_llama_matrix.py" --root "$root" --precision fp16 --model-path "$root/model"
    echo FP16_MATRIX_COMPLETE
fi
