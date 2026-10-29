#!/usr/bin/env bash
set -euo pipefail
repo=$1; root=$2
python=/data/users/shveikinda/miniconda3/bin/python
compat="$HOME/projects/low-bit-reasoning/exps/1029-gpqa-rescue-vv/compat"
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=GPU-81dfcc32-ea97-c0f1-778e-ff44c4992c1a
export HF_HOME=/data/hf_cache PYTHONPATH="$compat:$repo" WANDB_MODE=offline PYTHONUNBUFFERED=1
cd "$repo"
"$python" "$repo/vv/diagnose_distill_backends.py" --root "$root" --backend vllm
HF_HOME="$root/gptq-hf-cache" HF_MODULES_CACHE="$root/gptq-hf-cache/modules" \
    PYTHONPATH="$root/gptq-pydeps:$repo" "$python" "$repo/vv/diagnose_distill_backends.py" --root "$root" --backend gptq
echo BACKEND_DIAGNOSTICS_COMPLETE
