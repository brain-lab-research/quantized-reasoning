#!/usr/bin/env bash
# Take over only after the current FP16 process has exited; no trace is interrupted.
set -euo pipefail
active=$1
legacy_parent=$2
repo=$3
root=$4
shift 4
. "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate base
export CUDA_MPS_PIPE_DIRECTORY="$root/direct-cuda"
export WANDB_MODE=offline PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
while [[ "$active" != none ]] && kill -0 "$active" 2>/dev/null; do
    # A stopped parent cannot reap its finished GPU child until resumed.
    [[ $(ps -p "$active" -o stat=) == Z* ]] && break
    sleep 10
done
# The old parent is stopped and its GPU child has now finished. Terminate only
# that idle scheduler so its obsolete full-GSM8K backlog cannot start.
if [[ "$legacy_parent" != none ]]; then
    kill -TERM "$legacy_parent" 2>/dev/null || true
    kill -CONT "$legacy_parent" 2>/dev/null || true
fi
set -a
. "$HOME/.s3-cod.env"
set +a
sync_once() { /usr/bin/python3 "$repo/h200/sync.py" --root "$root" --prefix "shveikinda/vv/low-bit/$(basename "$root")"; }
(while sleep 180; do sync_once || printf 'BACKUP_RETRY\n'; done) &
sync_pid=$!
finish() { rc=$?; trap - EXIT; kill "$sync_pid" 2>/dev/null || true; sync_once || rc=1; exit "$rc"; }
trap finish EXIT
trap 'exit 143' TERM INT
cd "$repo"
python "$repo/h200/paper_matrix.py" --root "$root" "$@"
