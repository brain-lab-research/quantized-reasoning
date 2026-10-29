#!/usr/bin/env python3
"""Collect ordered single-model benchmark ranges with one shared vLLM engine."""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import deque
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def main():
    import yaml
    from transformers import AutoTokenizer
    from vllm import SamplingParams
    from vllm.sampling_params import RequestOutputKind
    from quantlab.actors.backends.vllm_backend import VLLMBackend
    from quantlab.artifacts.store import ArtifactStore
    from quantlab.benchmarks.registry import BenchmarkRegistry
    from quantlab.config.schema import ExperimentConfig
    from quantlab.core.trace import Trace, TraceSegment
    from quantlab.core.types import PrecisionMode, QuantizationMethod, SegmentRole
    from quantlab.metrics.registry import MetricRegistry
    from quantlab.runner import _load_benchmark_examples, _record_example
    from scripts.collect_baseline_traces import engine_prompt, format_native_prompts, set_requested_range

    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--model-path", required=True)
    p.add_argument("--max-tokens", type=int, default=32768)
    p.add_argument("--max-num-seqs", type=int, default=16)
    p.add_argument("--kv-cache-gib", type=float, default=80)
    p.add_argument("--memory-utilization", type=float, default=0.75)
    p.add_argument("--max-outstanding", type=int, default=32)
    p.add_argument("--shared-prefix", help="S3 queue prefix; optional for backward compatibility")
    p.add_argument("--shared-owner", help="Stable owner across retries")
    p.add_argument("--shared-fingerprint", help="Pinned model and decoding fingerprint")
    p.add_argument("--reverse", action="store_true", help="Consume queue from the end")
    p.add_argument("--ready-barrier", help="S3 key that releases this worker after model initialization")
    args = p.parse_args()
    if args.max_outstanding < args.max_num_seqs:
        p.error("max-outstanding must be at least max-num-seqs")
    coordinator = None
    if args.shared_prefix:
        if not args.shared_owner or not args.shared_fingerprint:
            p.error('shared-owner and shared-fingerprint are required')
        import boto3
        from scripts.shared_queue import SharedQueue
        client = boto3.client('s3', endpoint_url='https://s3.cod.phystech.edu')
        coordinator = SharedQueue(client, 'iii-storage2', args.shared_prefix,
                                  args.shared_owner, args.shared_fingerprint)
    tasks = json.loads(args.manifest.read_text())
    if not tasks:
        p.error("manifest must be nonempty")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    contexts = []
    signature = None
    for task in tasks:
        raw = yaml.safe_load(Path(task["config"]).read_text())
        if len(raw["actors"]) != 1 or len(raw["pipeline"]) != 1 or raw["pipeline"][0].get("exit_conditions"):
            raise ValueError("Only single-stage baselines without exit conditions are supported")
        set_requested_range(raw, int(task["count"]), int(task["start"]))
        raw["actors"][0]["generation_params"]["max_new_tokens"] = args.max_tokens
        raw["pipeline"][0]["max_new_tokens"] = args.max_tokens
        kwargs = raw["actors"][0]["backend_kwargs"]
        kwargs.pop("cuda_visible_devices", None)
        kwargs.update(max_num_seqs=args.max_num_seqs, gpu_memory_utilization=args.memory_utilization,
                      max_num_batched_tokens=8192,
                      max_model_len=max(kwargs.get("max_model_len", 0), args.max_tokens + 4096))
        if args.kv_cache_gib > 0:
            kwargs["kv_cache_memory_bytes"] = int(args.kv_cache_gib * 1024**3)
        else:
            kwargs.pop("kv_cache_memory_bytes", None)
        output = Path(task["output"])
        raw["output"].update(base_dir=str(output), staged_wave_checkpoints=False)
        raw["staged_execution"] = False
        raw["wandb"] = {"enabled": False}
        raw["description"] = raw.get("description", "") + " Prompt recipe: native_chat_explicit_token_ids_v2."
        config = ExperimentConfig.model_validate(raw)
        actor = config.actors[0]
        if coordinator:
            sampling = actor.generation_params.model_dump()
            expected = {'max_new_tokens': 32768, 'temperature': 0.6, 'top_p': 0.95,
                        'top_k': 20, 'repetition_penalty': 1.0,
                        'stop_sequences': ['<|im_end|>', '<|endoftext|>'], 'seed': 42}
            if (actor.quantization != 'gptq' or actor.quantization_config.get('bits') != 2
                    or actor.quantization_config.get('group_size') != 32
                    or any(sampling.get(k) != v for k, v in expected.items())):
                raise ValueError('Shared-queue model or sampling parameters differ')
        current = (actor.model_id, actor.precision, actor.quantization, actor.backend_kwargs)
        if signature is not None and current != signature:
            raise ValueError("A shared engine requires identical model and engine configuration")
        signature = current
        adapter = BenchmarkRegistry.build(config.benchmark.name)
        if config.benchmark.name == "gpqa_diamond":
            from scripts.measure_gpqa_rescue import paper_gpqa_choice
            adapter.extract_answer = paper_gpqa_choice
        examples = _load_benchmark_examples(adapter, config.benchmark)
        format_native_prompts(examples, tokenizer)
        output.mkdir(parents=True, exist_ok=True)
        store = ArtifactStore(str(output))
        pointer = output / "run-id.txt"
        if pointer.exists():
            run_id = pointer.read_text().strip()
            if json.loads((output / run_id / "config.json").read_text()) != config.model_dump():
                raise ValueError(f"Resume config mismatch: {output}")
        else:
            run_id = store.new_run(config.experiment_name, config.model_dump())
            pointer.write_text(run_id)
        saved = {r["example_id"]: r for r in store.load_judgements(run_id)}
        expected = {e.example_id for e in examples}
        if len(expected) != len(examples) or not set(saved).issubset(expected):
            raise ValueError(f"Invalid or duplicate example IDs: {output}")
        pending = deque(e for e in (reversed(examples) if args.reverse else examples) if e.example_id not in saved)
        gen = actor.generation_params
        params = SamplingParams(max_tokens=gen.max_new_tokens, temperature=gen.temperature,
                                top_p=gen.top_p, top_k=gen.top_k, seed=gen.seed,
                                stop=gen.stop_sequences or None, output_kind=RequestOutputKind.FINAL_ONLY)
        contexts.append(dict(config=config, adapter=adapter, store=store, run_id=run_id,
                             saved=saved, pending=pending, examples=examples, params=params,
                             metrics=[MetricRegistry.build(m.name, **m.kwargs) for m in config.metrics],
                             tokens=0, started=time.monotonic()))
        print(f"READY {config.benchmark.name} total={len(examples)} saved={len(saved)} pending={len(pending)}", flush=True)
    if args.reverse:
        contexts.reverse()
    if not any(c["pending"] for c in contexts):
        print("ALL_COMPLETE", flush=True)
        return
    actor = contexts[0]["config"].actors[0]
    backend = VLLMBackend(args.model_path, precision=PrecisionMode(actor.precision),
                          quantization=QuantizationMethod(actor.quantization), **actor.backend_kwargs)
    backend.load()
    if coordinator is not None:
        # Repair interrupted writes: republish records owned by this worker.
        for ctx in contexts:
            examples = {e.example_id: e for e in ctx['examples']}
            traces = {t.example_id: t for t in ctx['store'].load_traces(ctx['run_id'])}
            for eid, judgement in ctx['saved'].items():
                if eid not in traces:
                    raise ValueError(f'Missing trace for saved judgement: {eid}')
                example = examples[eid]
                etag = coordinator.reserve(ctx['config'].benchmark.name, eid, example.prompt)
                if etag:
                    coordinator.finish(ctx['config'].benchmark.name, eid, example.prompt, etag,
                                       {'trace': traces[eid].to_dict(), 'judgement': judgement})
    print('ENGINE_READY', flush=True)
    if args.ready_barrier:
        if coordinator is None:
            raise ValueError('ready-barrier requires a shared queue')
        while True:
            try:
                coordinator.client.head_object(Bucket=coordinator.bucket, Key=args.ready_barrier)
                break
            except Exception as exc:
                if getattr(exc, 'response', {}).get('Error', {}).get('Code') not in ('404', 'NoSuchKey', 'NotFound'):
                    raise
                time.sleep(2)
        print('BARRIER_RELEASED', flush=True)
    engine = backend._llm.llm_engine
    inflight = {}
    sequence = 0
    def admit():
        nonlocal sequence
        for ctx in contexts:
            while ctx["pending"] and len(inflight) < args.max_outstanding:
                example = ctx["pending"].popleft()
                etag = coordinator.reserve(ctx['config'].benchmark.name, example.example_id, example.prompt) if coordinator else None
                if coordinator and not etag:
                    continue
                request_id = f"multi-{sequence}"
                sequence += 1
                engine.add_request(request_id, engine_prompt(example.prompt, tokenizer, True), ctx["params"])
                inflight[request_id] = (ctx, example, etag)
            if len(inflight) >= args.max_outstanding:
                break
    try:
        admit()
        while inflight:
            for output in engine.step():
                if not output.finished:
                    continue
                if output.request_id not in inflight:
                    raise RuntimeError(f"Unknown or duplicate request: {output.request_id}")
                ctx, example, etag = inflight.pop(output.request_id)
                generated = output.outputs[0]
                trace = Trace(example.example_id, example.prompt)
                trace.append_segment(TraceSegment(
                    ctx["config"].actors[0].actor_id, generated.text, len(generated.token_ids), 0,
                    role=SegmentRole.ANSWER, llm_prompt_full=example.prompt,
                    metadata={"token_ids": list(generated.token_ids),
                              "prompt_token_ids": list(output.prompt_token_ids),
                              "finish_reason": generated.finish_reason, "serving_mode": "continuous_multi"},
                ))
                trace.finished_at = time.time()
                _record_example(store=ctx["store"], run_id=ctx["run_id"], config=ctx["config"],
                                metrics=ctx["metrics"], adapter=ctx["adapter"], example=example,
                                trace=trace, verbose=True)
                rows = ctx["store"].load_judgements(ctx["run_id"])
                combined = {r["example_id"]: r for r in rows}
                if coordinator:
                    coordinator.finish(ctx['config'].benchmark.name, example.example_id,
                                       example.prompt, etag,
                                       {'trace': trace.to_dict(), 'judgement': combined[example.example_id]})
                ctx["saved"] = combined
                ctx["tokens"] += len(generated.token_ids)
                summary = {"n_examples": len(ctx["examples"]), "n_judged": len(combined),
                           "n_new": len(combined), "n_errors": 0,
                           "accuracy": sum(r["is_correct"] for r in combined.values()) / len(combined),
                           "generated_tokens": ctx["tokens"],
                           "wall_seconds": time.monotonic() - ctx["started"]}
                ctx["store"].save_summary(ctx["run_id"], summary)
                print(f"SAVED {ctx['config'].benchmark.name} {example.example_id} "
                      f"total={len(combined)}/{len(ctx['examples'])}", flush=True)
            admit()
        if any(ctx["pending"] for ctx in contexts):
            raise RuntimeError("Queue ended with pending requests")
        print("LOCAL_QUEUE_EXHAUSTED" if coordinator else "ALL_COMPLETE", flush=True)
    finally:
        backend.unload()


if __name__ == "__main__":
    main()
