#!/usr/bin/env python3
"""Warm single-request wall-clock profiles, separate from model loading and grading."""
import argparse
import atexit
import json
import os
from pathlib import Path
import sys
import signal
import statistics
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    import yaml
    import torch
    import vllm
    from vllm import SamplingParams
    from quantlab.actors.backends.vllm_backend import VLLMBackend
    from quantlab.benchmarks.registry import BenchmarkRegistry
    from quantlab.config.schema import ExperimentConfig
    from quantlab.core.types import PrecisionMode, QuantizationMethod
    from quantlab.runner import _load_benchmark_examples

    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--examples", type=int, default=20)
    p.add_argument("--deadline", type=float, default=float("inf"), help="UTC Unix timestamp")
    args = p.parse_args()
    raw = yaml.safe_load(args.config.read_text())
    raw["benchmark"]["max_examples"] = args.examples
    actor = raw["actors"][0]
    actor["backend_kwargs"].pop("cuda_visible_devices", None)
    actor["backend_kwargs"].update(enforce_eager=False, max_num_seqs=1,
        max_num_batched_tokens=4096, max_model_len=36864, gpu_memory_utilization=0.8,
        kv_cache_memory_bytes=12 * 1024**3, enable_prefix_caching=True)
    config = ExperimentConfig.model_validate(raw)
    adapter = BenchmarkRegistry.build(config.benchmark.name)
    examples = _load_benchmark_examples(adapter, config.benchmark)
    for example in examples:
        if example.prompt.endswith("<|im_start|>assistant\n"):
            example.prompt += "<think>\n"
    args.output.mkdir(parents=True, exist_ok=True)
    cfg_path = args.output / "config.json"
    effective = config.model_dump()
    if cfg_path.exists() and json.loads(cfg_path.read_text()) != effective:
        raise ValueError("Profile resume config mismatch")
    cfg_path.write_text(json.dumps(effective, indent=2))
    path = args.output / "results.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []
    done = {r["example_id"] for r in rows}
    if len(done) != len(rows):
        raise ValueError("Duplicate profile results")
    pending = [e for e in examples if e.example_id not in done]
    if not pending or time.time() >= args.deadline:
        print("PROFILE_ALREADY_COMPLETE_OR_DEADLINE", flush=True)
        return
    actor = config.actors[0]
    backend = VLLMBackend(actor.model_id, precision=PrecisionMode(actor.precision),
                         quantization=QuantizationMethod(actor.quantization), **actor.backend_kwargs)
    atexit.register(lambda: backend.unload() if getattr(backend, "_llm", None) is not None else None)
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    started = time.perf_counter()
    backend.load()
    load_seconds = time.perf_counter() - started
    model = backend._llm
    warmup = ("<|im_start|>user\nCompute 2+3.<|im_end|>\n"
              "<|im_start|>assistant\n<think>\n\n</think>\n\n")
    warmup_started = time.perf_counter()
    model.generate([warmup], SamplingParams(temperature=0, max_tokens=32), use_tqdm=False)
    warmup_seconds = time.perf_counter() - warmup_started
    (args.output / "runtime.json").write_text(json.dumps(dict(
        vllm=vllm.__version__, torch=torch.__version__, model_load_seconds=load_seconds,
        warmup_seconds=warmup_seconds,
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        slurm_job_id=os.environ.get("SLURM_JOB_ID"), mode="sequential batch=1, warm, prefix caching enabled"), indent=2))
    gen = actor.generation_params
    params = SamplingParams(max_tokens=gen.max_new_tokens, temperature=gen.temperature,
        top_p=gen.top_p, top_k=gen.top_k, seed=gen.seed, stop=gen.stop_sequences or None)
    with path.open("a", buffering=1) as output_file:
        for example in pending:
            if time.time() >= args.deadline:
                break
            started = time.perf_counter()
            result = model.generate([example.prompt], params, use_tqdm=False)[0]
            elapsed = time.perf_counter() - started
            answer = result.outputs[0]
            predicted = adapter.extract_answer(answer.text)
            row = dict(example_id=example.example_id, wall_seconds=elapsed,
                prompt_tokens=len(result.prompt_token_ids), output_tokens=len(answer.token_ids),
                token_ids=list(answer.token_ids), text=answer.text, finish_reason=answer.finish_reason,
                predicted=predicted, is_correct=bool(adapter.is_correct(predicted, example.ground_truth)))
            output_file.write(json.dumps(row) + "\n")
            rows.append(row)
            print("PROFILE_SAVED", example.example_id, elapsed, len(answer.token_ids), flush=True)
            summary = dict(expected=len(examples), completed=len(rows),
                complete=len(rows) == len(examples), model_load_seconds=load_seconds,
                mean_wall_seconds=sum(r["wall_seconds"] for r in rows)/len(rows),
                total_output_tokens=sum(r["output_tokens"] for r in rows),
                mode="warm batch=1; excludes model load, warmup and answer grading")
            times = [r["wall_seconds"] for r in rows]
            summary["p50_wall_seconds"] = statistics.median(times)
            summary["p95_wall_seconds"] = statistics.quantiles(times, n=100, method="inclusive")[94] if len(times)>1 else times[0]
            (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    backend.unload()
    print("PROFILE_FINISHED", flush=True)


if __name__ == "__main__":
    main()
