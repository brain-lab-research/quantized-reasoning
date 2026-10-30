#!/usr/bin/env bash
set -euo pipefail
repo=$(cd "$(dirname "$0")/.." && pwd)
python="/home/data/$USER/envs/low-bit-vllm/bin/python"
root="/home/data/$USER/projects/low-bit-reasoning/exps/1030-plus-p-shared"
mkdir -p "$root/plans"
set -a; . "$HOME/.s3-cod.env"; set +a
export PYTHONPATH="$repo" WANDB_MODE=offline
"$python" - "$root/plans" <<'PY'
import hashlib,json,sys,boto3
from pathlib import Path
root=Path(sys.argv[1]); s=boto3.client('s3',endpoint_url='https://s3.cod.phystech.edu')
bucket='iii-storage2'; prefix='shveikinda/vv/1030-distill-plus-p/'
manifest=json.loads(s.get_object(Bucket=bucket,Key=prefix+'plans-manifest.json')['Body'].read())
raw=json.dumps(manifest,sort_keys=True,separators=(',',':')).encode()
assert hashlib.sha256(raw).hexdigest()[:24]=='21e432748b35b06a5df3fd2d'
assert len(manifest)==9 and sum(x['count'] for x in manifest.values())==9387
for name,info in manifest.items():
    if Path(name).name != name or not name.endswith('.jsonl'): raise ValueError(name)
    path=root/name
    if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest()!=info['sha256']:
        temp=path.with_suffix('.tmp')
        s.download_file(bucket,prefix+'plans/'+name,str(temp))
        if hashlib.sha256(temp.read_bytes()).hexdigest()!=info['sha256']: raise ValueError('PLAN_SHA_MISMATCH '+name)
        temp.replace(path)
    assert sum(1 for _ in path.open())==info['count']
print('PLANS_VERIFIED 9387',flush=True)
PY
for owner in fpmi-a fpmi-b; do
    [[ ! -e "$root/jobs-$owner.txt" ]] || { echo "ALREADY_QUEUED $owner"; exit 1; }
    "$python" "$repo/scripts/prepare_plus_p_manifest.py" --root "$root/$owner"
done
created=()
rollback() { rc=$?; trap - EXIT; if ((rc != 0)); then for id in "${created[@]}"; do scancel "$id" 2>/dev/null || true; done; fi; exit "$rc"; }
trap rollback EXIT
for owner in fpmi-a fpmi-b; do
    prev=''; ids="$root/jobs-$owner.txt"
    for idx in 0 1 2; do
        opts=()
        if [[ -z "$prev" ]]; then opts+=(--hold); else opts+=(--dependency="afterany:$prev"); fi
        job=$(sbatch --parsable -A fpmi-p10 -p gpu --gres=gpu:1 --cpus-per-task=8 --mem=84G --time=08:00:00 \
            "${opts[@]}" --job-name="plus-p-${owner#fpmi-}-$idx" --output="$root/$owner/worker-$idx.log" \
            "$repo/h200/plus_p_shared_worker.sh" "$repo" "$root" "$owner")
        created+=("$job"); printf '%s\n' "$job" >> "$ids"; prev=$job
    done
done
trap - EXIT
# Pause only this project's previously queued W4 jobs before cancelling active W4 work.
while IFS='|' read -r id name state; do
    case "$name" in w4-post-*|w4-after-*|w4-b32-*|w4-32-*)
        if [[ "$state" == PENDING ]]; then scontrol hold "$id"; echo W4_HELD "$id" "$name"; fi;; esac
done < <(squeue -h -u "$USER" -o '%i|%j|%T')
while IFS='|' read -r id name state; do
    case "$name" in w4-post-*|w4-after-*|w4-b32-*|w4-32-*)
        if [[ "$state" == RUNNING ]]; then scancel "$id"; echo W4_STOPPED "$id" "$name"; fi;; esac
done < <(squeue -h -u "$USER" -o '%i|%j|%T')
for owner in fpmi-a fpmi-b; do
    first=$(head -n 1 "$root/jobs-$owner.txt"); scontrol release "$first"
    echo "PLUS_P_$owner $(tr '\n' ' ' < "$root/jobs-$owner.txt")"
done
git -C "$repo" log -1 --oneline
squeue -u "$USER" -o '%.8i %.22j %.9T %.10M %.25R'
