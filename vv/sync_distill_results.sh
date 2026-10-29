#!/usr/bin/env bash
set -euo pipefail
repo=$1; root=$2
python=/data/users/shveikinda/miniconda3/bin/python
export PYTHONPATH="$root/s3-pydeps" AWS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt
set -a; . "$HOME/.s3-cod.env"; set +a
while true; do
    total=0
    for dir in "$root"/calibration "$root"/1029-r1-distill-llama-8b-*-fp16 "$root"/1029-r1-distill-llama-8b-*-w2; do
        [[ -d "$dir" ]] || continue
        bytes=$(du -sb "$dir" | awk '{print $1}')
        total=$((total + bytes))
    done
    if (( total > 2 * 1024 * 1024 * 1024 )); then
        echo BACKUP_CAP_EXCEEDED "$total"; exit 1
    fi
    for dir in "$root"/calibration "$root"/1029-r1-distill-llama-8b-*-fp16 "$root"/1029-r1-distill-llama-8b-*-w2; do
        [[ -d "$dir" ]] || continue
        "$python" "$repo/h200/sync.py" --root "$dir" --prefix "shveikinda/vv/$(basename "$root")/$(basename "$dir")" || echo BACKUP_RETRY
    done
    echo "SYNC_PASS total=$total"
    sleep 180
done
