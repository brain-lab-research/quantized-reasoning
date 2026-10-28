#!/usr/bin/env bash
# A successor, not a parallel baseline worker: wait until the current controller exits.
set -euo pipefail
repo=$1
primary=$2
root=$3
controller=$4
deadline=$5
gpu_index=$6
expected_uuid=$7
actual_uuid=$(/usr/bin/nvidia-smi -i "$gpu_index" --query-gpu=uuid --format=csv,noheader | tr -d ' \r')
[[ "$actual_uuid" == "$expected_uuid" ]] || { echo "GPU_UUID_MISMATCH"; exit 1; }
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$gpu_index"
export HF_HOME=/data/hf_cache CUDA_MPS_PIPE_DIRECTORY="$root/direct-cuda"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false WANDB_MODE=offline
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 VLLM_WORKER_MULTIPROC_METHOD=spawn
. "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate base
mkdir -p "$root"
while kill -0 "$controller" 2>/dev/null; do
    command=$(tr '\0' ' ' < "/proc/$controller/cmdline" 2>/dev/null || true)
    [[ "$command" == *paper_matrix.py* && "$command" == *"$primary"* ]] || break
    (( $(date +%s) < deadline )) || exit 0
    printf 'WAIT_PRIMARY pid=%s\n' "$controller"
    sleep 15
done
printf 'PRIMARY_EXITED\n'
remaining=$(( deadline - $(date +%s) ))
(( remaining > 0 )) || exit 0
# Retry only incomplete primary tasks before spending GPU time on follow-ups.
if ! timeout --signal=TERM --kill-after=30s "$remaining" bash "$primary/cmd.txt" > "$root/baseline-check.log" 2>&1; then
    printf 'BASELINE_RETRY_FAILED: continuing independent follow-ups\n'
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
remaining=$(( deadline - $(date +%s) ))
grade_pid=''
if (( remaining > 3600 )); then
    printf 'ROUTING_CONTROL_START\n'
    if timeout --signal=TERM --kill-after=30s "$remaining" python "$repo/scripts/run_fp16_recovery.py" \
        --input "$root/1028-routing-control/requests.jsonl" --output "$root/1028-routing-control" \
        --model Qwen/Qwen3-32B --kv-cache-gib 56 --max-num-seqs 64 --compiled \
        > "$root/1028-routing-control/run.log" 2>&1; then
        timeout 3600 python "$repo/scripts/grade_routing_control.py" --root "$root/1028-routing-control" \
            --source "$root/source-w2" > "$root/1028-routing-control/grade.log" 2>&1 &
        grade_pid=$!
    else
        printf 'ROUTING_CONTROL_FAILED: continuing latency profiles\n'
    fi
fi
for pair in 32:fp16 32:w2 8:fp16 8:w2; do
    remaining=$(( deadline - $(date +%s) ))
    (( remaining > 600 )) || break
    size=${pair%:*}
    precision=${pair#*:}
    out="$root/1028-latency-qwen${size}b-$precision"
    mkdir -p "$out"
    cfg="$primary/1027-paper-qwen${size}b-math500-${precision}-all/cfg.yaml"
    printf 'LATENCY_START %s\n' "$pair"
    if ! timeout --signal=TERM --kill-after=30s "$remaining" python "$repo/scripts/profile_baseline_latency.py" \
        --config "$cfg" --output "$out" --examples 20 --deadline "$deadline" > "$out/run.log" 2>&1; then
        printf 'LATENCY_FAILED %s\n' "$pair"
    fi
done
if [[ -n "$grade_pid" ]]; then wait "$grade_pid" || printf 'ROUTING_GRADE_FAILED\n'; fi
printf 'OVERNIGHT_FOLLOWUPS_FINISHED\n'
