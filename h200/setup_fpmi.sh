#!/usr/bin/env bash
# Run inside tmux in the portal. Uses the explicitly approved 390 GiB ceiling.
set -euo pipefail
repo=$(cd "$(dirname "$0")/.." && pwd)
root="/home/data/$USER/projects/low-bit-reasoning/exps/1027-fpmi-w2-traces"
base="/home/data/$USER/projects/low-bit-reasoning/runtime"
ql_env="/home/data/$USER/envs/low-bit-vllm"
ql_python="$ql_env/bin/python"
mkdir -p "$root" "$base" "/home/data/$USER/tmp"
export TMPDIR="/home/data/$USER/tmp" UV_NO_CACHE=1 UV_CONCURRENT_DOWNLOADS=4
used=$(du -sk "/home/data/$USER" | cut -f1)
(( used + 25*1024*1024 < 390*1024*1024 )) || { echo DISK_HEADROOM_FAILED; exit 1; }
if [[ ! -x "$ql_python" ]]; then
    uv venv --python "/home/data/$USER/envs/codi/bin/python" "$ql_env"
fi
uv pip install --python "$ql_python" --no-cache 'vllm==0.19.0' 'transformers==4.57.1' 'datasets==4.8.4' pandas pyarrow click boto3 rich PyYAML accelerate
export PYTHONPATH="$repo" PATH="$ql_env/bin:$PATH"
"$ql_python" -c 'import torch,vllm,transformers; print("STACK_READY",torch.__version__,vllm.__version__,transformers.__version__)'
export HF_HOME="/home/data/$USER/hf-cache"
"$ql_python" -c 'from huggingface_hub import snapshot_download; print(snapshot_download("kaitchup/Qwen3-8B-autoround-2bit-gptq",allow_patterns=["*.json","*.safetensors","*.model","tokenizer*"],max_workers=4))'
set -a
. "$HOME/.s3-cod.env"
set +a
"$ql_python" "$repo/h200/pull_trace_reuse.py" --root "$root"
used=$(du -sk "/home/data/$USER" | cut -f1)
(( used < 390*1024*1024 )) || { echo DISK_CEILING_FAILED; exit 1; }
"$ql_python" "$repo/h200/paper_matrix.py" --root "$root" --prepare-only --data-smoke --sizes 8 --precisions w2
smoke=$(sbatch --parsable -A fpmi-p10 -p gpu --gres=gpu:1 --cpus-per-task=8 --mem=64G --time=00:10:00 --job-name=lowbit-smoke --output="$root/smoke.log" "$repo/h200/fpmi_worker.sh" "$repo" "$root" smoke)
job=$(sbatch --parsable -A fpmi-p10 -p gpu --gres=gpu:1 --cpus-per-task=8 --mem=64G --time=08:00:00 --dependency="afterok:$smoke" --job-name=lowbit-w2 --output="$root/run.log" "$repo/h200/fpmi_worker.sh" "$repo" "$root" full)
printf '%s\n' "$smoke" > "$root/smoke-job.txt"
printf '%s\n' "$job" > "$root/job-id.txt"
next=$(sbatch --parsable -A fpmi-p10 -p gpu --gres=gpu:1 --cpus-per-task=8 --mem=64G --time=08:00:00 --dependency="afterany:$job" --job-name=lowbit-w2-resume --output="$root/resume.log" "$repo/h200/fpmi_worker.sh" "$repo" "$root" full)
printf '%s\n' "$next" > "$root/resume-job.txt"
printf 'SMOKE=%s FULL=%s RESUME=%s\n' "$smoke" "$job" "$next"
squeue -j "$smoke,$job,$next" -o '%.8i %.12j %.9T %.20b %.30R'
