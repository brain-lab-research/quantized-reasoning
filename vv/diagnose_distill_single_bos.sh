#!/usr/bin/env bash
set -uo pipefail
repo=$1; root=$2
python=/data/users/shveikinda/miniconda3/bin/python
compat="$HOME/projects/low-bit-reasoning/exps/1029-gpqa-rescue-vv/compat"
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=GPU-81dfcc32-ea97-c0f1-778e-ff44c4992c1a
export HF_HOME=/data/hf_cache PYTHONPATH="$compat:$repo" WANDB_MODE=offline PYTHONUNBUFFERED=1
cd "$repo" || exit 1
"$python" "$repo/vv/smoke_distill_w2_quality.py" --root "$root" --precision w2
w2_status=$?
"$python" "$repo/vv/smoke_distill_w2_quality.py" --root "$root" --precision fp16
fp16_status=$?
echo "SINGLE_BOS_DIAGNOSTIC_FINISHED w2=$w2_status fp16=$fp16_status"
exit $((w2_status || fp16_status))
