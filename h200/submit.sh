#!/usr/bin/env bash
# Usage: bash h200/submit.sh /absolute/path/to/existing/env/bin/python
set -euo pipefail
ql_python=$1
repo=$(cd "$(dirname "$0")/.." && pwd)
export QL_SERIES=${QL_SERIES:-$(date +%m%d)-rebuttal-traces}
root="/home/data/$USER/projects/low-bit-reasoning/exps/$QL_SERIES"
mkdir -p "$root"
"$ql_python" -c 'import torch,vllm,transformers,pydantic,yaml,datasets,click; print("STACK",torch.__version__,vllm.__version__,transformers.__version__)'
test -f "$HOME/.s3-cod.env"
# Above the soft limit, first clean already-backed-up artifacts explicitly.
used_kib=$(du -sk "/home/data/$USER" | cut -f1)
if (( used_kib > 200*1024*1024 )); then
    printf 'DISK_GATE: %s KiB exceeds 200 GiB soft limit; clean verified backups first.\n' "$used_kib" >&2
    exit 1
fi
if [[ -f "$root/job-id.txt" ]]; then
    existing=$(cat "$root/job-id.txt")
    if squeue -h -j "$existing" | grep -q .; then
        printf 'ALREADY_SUBMITTED %s\n' "$existing"
        squeue -j "$existing"
        exit 0
    fi
fi
printf 'sbatch -A fpmi-p10 -p gpu --gres=gpu:1 --time=08:00:00 --cpus-per-task=8 --mem=64G --output=%q %q %q %q %q\n' "$root/run.log" "$repo/h200/job.sh" "$repo" "$root" "$ql_python" > "$root/cmd.txt"
job=$(sbatch --parsable -A fpmi-p10 -p gpu --gres=gpu:1 --time=08:00:00 --cpus-per-task=8 --mem=64G --output="$root/run.log" --job-name=lowbit-traces "$repo/h200/job.sh" "$repo" "$root" "$ql_python")
printf '%s\n' "$job" > "$root/job-id.txt"
git -C "$repo" log -1 --oneline
printf 'SUBMITTED %s ROOT %s\n' "$job" "$root"
squeue -j "$job" -o '%.8i %.9T %.20b %.30R'
