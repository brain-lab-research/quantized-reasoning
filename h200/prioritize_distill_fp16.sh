#!/usr/bin/env bash
# Move existing Distill jobs ahead of W4; preserve W4 artifacts and queue its continuation.
set -euo pipefail
repo=$(cd "$(dirname "$0")/.." && pwd)
w4="/home/data/$USER/projects/low-bit-reasoning/exps/1029-w4-32b-full"
distill="/home/data/$USER/projects/low-bit-reasoning/exps/1029-distill-llama-fp16-full"
for f in "$w4/after-distill-gate.txt" "$w4/jobs-a-after-distill.txt" "$w4/jobs-b-after-distill.txt"; do
    [[ ! -e "$f" ]] || { echo "ALREADY_PRIORITIZED $f"; exit 1; }
done
for f in "$w4/jobs-a-batch32.txt" "$w4/jobs-b-batch32.txt" "$distill/jobs-a.txt" "$distill/jobs-b.txt" "$distill/gate-job.txt"; do [[ -f "$f" ]] || { echo "MISSING $f" >&2; exit 1; }; done
first_a=$(head -n 1 "$distill/jobs-a.txt"); first_b=$(head -n 1 "$distill/jobs-b.txt")
last_a=$(tail -n 1 "$distill/jobs-a.txt"); last_b=$(tail -n 1 "$distill/jobs-b.txt")
old_gate=$(cat "$distill/gate-job.txt")
for job in "$first_a" "$first_b" "$last_a" "$last_b" "$old_gate"; do [[ "$job" =~ ^[0-9]+$ ]] || exit 1; done
for job in "$first_a" "$first_b"; do
    [[ "$(squeue -h -j "$job" -o '%T')" == PENDING ]] || { echo "DISTILL_NOT_PENDING $job" >&2; exit 1; }
done
created=(); held=()
rollback() {
    rc=$?; trap - EXIT
    if ((rc != 0)); then
        for job in "${created[@]}"; do scancel "$job" 2>/dev/null || true; done
        for job in "${held[@]}"; do scontrol update JobId="$job" Dependency="afterok:$old_gate" 2>/dev/null || true; scontrol release "$job" 2>/dev/null || true; done
        rm -f -- "$w4/after-distill-gate.txt" "$w4/jobs-a-after-distill.txt" "$w4/jobs-b-after-distill.txt"
    fi
    exit "$rc"
}
trap rollback EXIT
gate=$(sbatch --parsable --hold -A fpmi-p10 -p cpu --cpus-per-task=1 --mem=2G --time=00:10:00 \
    --dependency="afterany:$last_a:$last_b" --job-name=w4-after-distill-gate \
    --output="$w4/after-distill-gate.log" "$repo/h200/distill_complete_gate.sh" "$repo" "$distill")
created+=("$gate"); printf '%s\n' "$gate" > "$w4/after-distill-gate.txt"
for worker in a b; do
    previous=''; ids="$w4/jobs-$worker-after-distill.txt"
    for idx in 0 1 2; do
        if [[ -z "$previous" ]]; then dependency="afterok:$gate"; else dependency="afterany:$previous"; fi
        memory=96G; [[ "$worker" == b ]] && memory=84G
        job=$(sbatch --parsable -A fpmi-p10 -p gpu --gres=gpu:1 --cpus-per-task=8 --mem="$memory" --time=08:00:00 \
            --dependency="$dependency" --export="ALL,LOWBIT_W4_SEQS_32=32,W4_JOB_IDS_FILE=$ids" \
            --job-name="w4-after-$worker-$idx" --output="$w4/worker-$worker-after-distill-$idx.log" \
            "$repo/h200/w4_matrix_worker.sh" "$repo" "$w4" "$worker")
        created+=("$job"); printf '%s\n' "$job" >> "$ids"; previous=$job
    done
done
# Hold Llama until old W4 workers stop, and make both RAM requests fit 181 GiB.
for job in "$first_a" "$first_b"; do scontrol hold "$job"; held+=("$job"); done
while read -r job; do [[ "$job" =~ ^[0-9]+$ ]] && scontrol update JobId="$job" MinMemoryNode=86016; done < "$distill/jobs-b.txt"
running=()
for worker in a b; do
    while read -r job; do
        [[ "$job" =~ ^[0-9]+$ ]] || { echo INVALID_W4_ID >&2; exit 1; }
        state=$(squeue -h -j "$job" -o '%T' 2>/dev/null || true)
        if [[ "$state" == RUNNING || "$state" == SUSPENDED || "$state" == COMPLETING ]]; then running+=("$job"); fi
    done < "$w4/jobs-$worker-batch32.txt"
done
dependency=''
if ((${#running[@]})); then joined=$(IFS=:; echo "${running[*]}"); dependency="afterany:$joined"; fi
for job in "$first_a" "$first_b"; do scontrol update JobId="$job" Dependency="$dependency"; done
# From here on, preserve the new continuation even if a later Slurm operation fails.
trap - EXIT
for worker in a b; do
    while read -r job; do [[ "$job" =~ ^[0-9]+$ ]] && scancel "$job" 2>/dev/null || true; done < <(tail -n +2 "$w4/jobs-$worker-batch32.txt")
done
for worker in a b; do
    job=$(head -n 1 "$w4/jobs-$worker-batch32.txt")
    [[ "$job" =~ ^[0-9]+$ ]] && scancel "$job" 2>/dev/null || true
done
scancel "$old_gate" 2>/dev/null || true
scontrol release "$gate"
for job in "$first_a" "$first_b"; do scontrol release "$job"; done
echo "DISTILL_FIRST=$first_a,$first_b W4_AFTER_GATE=$gate"
for worker in a b; do echo "W4_$worker $(tr '\n' ' ' < "$w4/jobs-$worker-after-distill.txt")"; done
squeue -u "$USER" -o '%.8i %.20j %.9T %.10M %.10m %.25R'
