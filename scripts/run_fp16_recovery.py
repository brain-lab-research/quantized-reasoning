#!/usr/bin/env python3
"""Paired FP16 recovery on a single model/backend, with immediate output saves."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.collect_baseline_traces import completed_outputs


def main():
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import RequestOutputKind
    import vllm
    import torch
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--kv-cache-gib", type=float, default=20)
    p.add_argument("--max-num-seqs", type=int, default=16)
    args = p.parse_args()
    source = [json.loads(l) for l in args.input.read_text().splitlines() if l.strip()]
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / "results.jsonl"
    completed = set()
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                completed.add(json.loads(line)["request_id"])
            except (json.JSONDecodeError, KeyError):
                pass
    pending = {r["request_id"]: r for r in source if r["request_id"] not in completed}
    print(f"RECOVERY pending={len(pending)} saved={len(completed)}", flush=True)
    if not pending:
        return
    config = {"model": args.model, "vllm": vllm.__version__, "torch": torch.__version__,
        "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "max_num_seqs": args.max_num_seqs, "kv_cache_gib": args.kv_cache_gib,
        "temperature": 0.6, "top_p": 0.95, "top_k": 20, "max_context": 36864,
        "timings": "paired arms on same backend; not comparable to NVIDIA latency"}
    (args.output / "config.json").write_text(json.dumps(config, indent=2))
    model = LLM(model=args.model, dtype="float16", tensor_parallel_size=1,
        enforce_eager=True, max_model_len=36864, gpu_memory_utilization=0.9,
        kv_cache_memory_bytes=int(args.kv_cache_gib * 1024**3),
        max_num_seqs=args.max_num_seqs, max_num_batched_tokens=4096)
    engine = model.llm_engine
    for rid, row in pending.items():
        engine.add_request(rid, row["prompt"], SamplingParams(
            temperature=0.6, top_p=0.95, top_k=20, seed=row["seed"],
            max_tokens=row["max_new_tokens"], output_kind=RequestOutputKind.FINAL_ONLY))
    started = time.monotonic()
    with path.open("a", buffering=1) as f:
        for output in completed_outputs(engine):
            row = pending[output.request_id]
            generated = output.outputs[0]
            record = {k: v for k, v in row.items() if k != "prompt"}
            record.update(text=generated.text, token_count=len(generated.token_ids),
                          token_ids=list(generated.token_ids), finish_reason=generated.finish_reason,
                          prompt_tokens=len(output.prompt_token_ids), wall_since_start=time.monotonic()-started)
            f.write(json.dumps(record) + "\n")
            print("SAVED", output.request_id, record["token_count"], flush=True)
    print("RECOVERY_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
