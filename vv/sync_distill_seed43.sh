#!/usr/bin/env bash
set -euo pipefail
repo=$1
session=${2:-lowbit-r1-seed43}
root="$HOME/projects/low-bit-reasoning/exps/1030-distill-w2-seed43"
set -a; . "$HOME/.s3-cod.env"; set +a
export PYTHONPATH="$HOME/projects/low-bit-reasoning/exps/1029-r1-distill-llama-8b/s3-pydeps"
sync_once() { /usr/bin/python3 "$repo/h200/sync.py" --root "$root" --prefix shveikinda/vv/low-bit/1030-distill-w2-seed43; }
while tmux has-session -t "=$session" 2>/dev/null; do sync_once || echo BACKUP_RETRY; sleep 180; done
sync_once
