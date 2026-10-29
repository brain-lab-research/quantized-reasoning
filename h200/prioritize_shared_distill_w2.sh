#!/usr/bin/env bash
set -euo pipefail
repo=$(cd "$(dirname "$0")/.." && pwd)
data_root=${LOWBIT_FPMI_DATA_ROOT:-"/home/data/$USER"}
shared="$data_root/projects/low-bit-reasoning/exps/1029-distill-shared-w2"
w4="$data_root/projects/low-bit-reasoning/exps/1029-w4-32b-full"
for f in "$shared/gate-job.txt" "$shared/jobs-a.txt" "$shared/jobs-b.txt" "$shared/audit-jobs.txt" "$w4/jobs-a-after-distill.txt" "$w4/jobs-b-after-distill.txt"; do
    [[ -f "$f" ]] || { echo "MISSING $f" >&2; exit 1; }
done
for f in "$shared/priority-gate-job.txt" "$w4/after-shared-gate.txt" "$w4/jobs-a-after-shared.txt" "$w4/jobs-b-after-shared.txt"; do
    [[ ! -e "$f" ]] || { echo "ALREADY_PRIORITIZED $f" >&2; exit 1; }
done
old_gate=$(cat "$shared/gate-job.txt")
first_a=$(head -n 1 "$shared/jobs-a.txt"); first_b=$(head -n 1 "$shared/jobs-b.txt")
audit_first=$(head -n 1 "$shared/audit-jobs.txt"); audit_last=$(tail -n 1 "$shared/audit-jobs.txt")
for job in "$old_gate" "$first_a" "$first_b" "$audit_first" "$audit_last"; do [[ "$job" =~ ^[0-9]+$ ]] || exit 1; done
for job in "$first_a" "$first_b"; do
    [[ "$(squeue -h -j "$job" -o '%T')" == PENDING ]] || { echo "SHARED_WORKER_NOT_PENDING $job" >&2; exit 1; }
done
created=(); held=()
rollback() {
    rc=$?; trap - EXIT
    if ((rc != 0)); then
        for job in "${held[@]}"; do scontrol update JobId="$job" Dependency="afterok:$old_gate" 2>/dev/null || true; scontrol release "$job" 2>/dev/null || true; done
        for job in "${created[@]}"; do scancel "$job" 2>/dev/null || true; done
        rm -f -- "$shared/priority-gate-job.txt" "$w4/after-shared-gate.txt" "$w4/jobs-a-after-shared.txt" "$w4/jobs-b-after-shared.txt"
    fi
    exit "$rc"
}
trap rollback EXIT
llama_gate=$(sbatch --parsable --hold -A fpmi-p10 -p cpu --cpus-per-task=1 --mem=2G --time=08:00:00 \
    --job-name=shared-w2-priority --output="$shared/priority-gate.log" "$repo/h200/shared_distill_w2_gate.sh" "$repo" priority)
created+=("$llama_gate"); printf '%s\n' "$llama_gate" > "$shared/priority-gate-job.txt"
w4_gate=$(sbatch --parsable --hold -A fpmi-p10 -p cpu --cpus-per-task=1 --mem=2G --time=00:10:00 \
    --dependency="afterany:$audit_first:$audit_last" --job-name=w4-after-shared-gate \
    --output="$w4/after-shared-gate.log" "$repo/h200/w4_after_shared_gate.sh")
created+=("$w4_gate"); printf '%s\n' "$w4_gate" > "$w4/after-shared-gate.txt"
for worker in a b; do
    previous=''; ids="$w4/jobs-$worker-after-shared.txt"
    for idx in 0 1 2; do
        if [[ -z "$previous" ]]; then dependency="afterok:$w4_gate"; else dependency="afterany:$previous"; fi
        job=$(sbatch --parsable -A fpmi-p10 -p gpu --gres=gpu:1 --cpus-per-task=8 --mem=84G --time=08:00:00 \
            --dependency="$dependency" --export="ALL,LOWBIT_W4_SEQS_32=32,W4_JOB_IDS_FILE=$ids" \
            --job-name="w4-post-$worker-$idx" --output="$w4/worker-$worker-after-shared-$idx.log" \
            "$repo/h200/w4_matrix_worker.sh" "$repo" "$w4" "$worker")
        created+=("$job"); printf '%s\n' "$job" >> "$ids"; previous=$job
    done
done
for job in "$first_a" "$first_b"; do scontrol hold "$job"; held+=("$job"); done
for job in "$first_a" "$first_b"; do scontrol update JobId="$job" Dependency="afterok:$llama_gate"; done
# From this point preserve the new queues even if a later Slurm operation fails.
trap - EXIT
scancel "$old_gate" 2>/dev/null || true
for worker in a b; do
    while read -r job; do [[ "$job" =~ ^[0-9]+$ ]] && scancel "$job" 2>/dev/null || true; done < <(tail -n +2 "$w4/jobs-$worker-after-distill.txt")
done
for worker in a b; do
    job=$(head -n 1 "$w4/jobs-$worker-after-distill.txt")
    [[ "$job" =~ ^[0-9]+$ ]] && scancel "$job" 2>/dev/null || true
done
scontrol release "$w4_gate"
scontrol release "$llama_gate"
for job in "$first_a" "$first_b"; do scontrol release "$job"; done
echo "DISTILL_PRIORITY_GATE=$llama_gate W4_AFTER_SHARED_GATE=$w4_gate"
for worker in a b; do echo "W4_$worker $(tr '\n' ' ' < "$w4/jobs-$worker-after-shared.txt")"; done
squeue -u "$USER" -o '%.8i %.22j %.9T %.10M %.10m %.25R'
