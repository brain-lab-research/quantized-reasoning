#!/usr/bin/env bash
# 128 C4 + 128 verified reasoning sequences, then GPTQ smoke and full runs.
set -euo pipefail
repo=$1; root=$2
python=/data/users/shveikinda/miniconda3/bin/python
compat="$HOME/projects/low-bit-reasoning/exps/1029-gpqa-rescue-vv/compat"
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=GPU-81dfcc32-ea97-c0f1-778e-ff44c4992c1a
export HF_HOME=/data/hf_cache PYTHONPATH="$compat:$repo" WANDB_MODE=offline PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false VLLM_WORKER_MULTIPROC_METHOD=spawn
cd "$repo"
[[ -f "$root/calibration/c4-windows.jsonl" && -f "$root/training_candidates.jsonl" ]] || { echo MISSING_CALIBRATION; exit 1; }
free=$(nvidia-smi -i 7 --query-gpu=memory.free --format=csv,noheader,nounits)
(( free >= 125000 )) || { echo GPU7_NOT_FREE "$free"; exit 1; }
if [[ ! -f "$root/calibration/teacher-windows.jsonl" ]]; then
    "$python" "$repo/vv/prepare_distill_teacher.py" --root "$root" --count 128
fi
if [[ ! -f "$root/model-gptq2-g32/calibration-provenance.json" ]]; then
    HF_HOME="$root/gptq-hf-cache" HF_MODULES_CACHE="$root/gptq-hf-cache/modules" \
        PYTHONPATH="$root/gptq-pydeps:$repo" "$python" "$repo/vv/quantize_distill_gptq2.py" \
        --root "$root" --c4-count 128 --teacher-count 128
fi
"$python" "$repo/vv/smoke_distill_w2_quality.py" --root "$root"
"$python" "$repo/vv/distill_llama_matrix.py" --root "$root" --prepare --precision w2
"$python" "$repo/vv/distill_llama_matrix.py" --root "$root" --precision w2 --model-path "$root/model-gptq2-g32"
echo W2_MATRIX_COMPLETE
"$python" "$repo/vv/distill_llama_matrix.py" --root "$root" --prepare --precision fp16
"$python" "$repo/vv/distill_llama_matrix.py" --root "$root" --precision fp16 --model-path "$root/model"
echo FP16_MATRIX_COMPLETE
